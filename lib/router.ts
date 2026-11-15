// Routing: which provider and model serve one request, and what happens when they refuse.
//
// Candidates are the quality table's free models that meet the request's requirements, best first:
// rating desc, then the provider's budget-range midpoint desc (a wide budget is likelier to still
// hold calls), then a provider on the client's own wire, then ids. One more filter stands above the
// sort: a client request in JSON mode never routes to an anthropic-wire provider, because that wire
// echoes json_mode false whatever it is sent, and the request would be lost on the echo check.
//
// Providers are spent as the trial goes. The entry tracks, per provider, the calls the env will
// have counted (a call naming a free model with the right key counts, a refusal on the schedule
// counts too) and the refusals it has given, in a row and in all. A provider is demoted to last
// resort — tried only when nothing live remains — once its counted calls reach the top of its
// published budget range, or once it has refused 3 times: the generator deals at most 2 scheduled
// refusals, so a third refusal, in a row or not, can only be a spent budget or an outage that has
// begun, and both are permanent. Calls are serial: a provider serves one at a time, and the client
// sends one request at a time, so the entry never holds two calls open on one provider.
//
// A refusal (402, 429, 5xx) is never the answer: the request walks to the next candidate. Exactly
// one successful provider call answers each request.

import {
  authHeaders,
  joinBase,
  type Brief,
  type Json,
  type Probe,
  type ProviderEntry,
  type Requirement,
} from './brief.ts';
import { buildUpstream, parseAnswer, type ClientView, type Normalized } from './wire.ts';

/** The published per-trial budget ranges (the kit's sealed table), by provider id. */
const BUDGET_RANGE: Record<string, { min: number; max: number }> = {
  groq: { min: 9, max: 12 },
  cerebras: { min: 5, max: 6 },
  gemini: { min: 6, max: 8 },
  zai: { min: 4, max: 5 },
  openrouter: { min: 10, max: 12 },
  sambanova: { min: 4, max: 5 },
  pollinations: { min: 10, max: 14 },
  cloudflare: { min: 7, max: 8 },
  ovh: { min: 5, max: 6 },
  kilol: { min: 12, max: 16 },
};

function budgetMid(providerId: string): number {
  const range = BUDGET_RANGE[providerId];
  return range === undefined ? 0 : (range.min + range.max) / 2;
}

function budgetTop(providerId: string): number {
  const range = BUDGET_RANGE[providerId];
  return range === undefined ? Number.POSITIVE_INFINITY : range.max;
}

function meetsRequirements(row: { tools: boolean; vision: boolean; json: boolean; context_length: number | null }, requires: Requirement): boolean {
  if (requires.tools === true && !row.tools) return false;
  if (requires.vision === true && !row.vision) return false;
  if (requires.json === true && !row.json) return false;
  if (requires.min_context !== undefined && (row.context_length === null || row.context_length < requires.min_context)) return false;
  return true;
}

function compareStrings(a: string, b: string): number {
  return a < b ? -1 : a > b ? 1 : 0;
}

export interface Candidate {
  provider: ProviderEntry;
  model: string;
  rating: number;
}

type Attempt =
  | { kind: 'answer'; answer: Normalized }
  | { kind: 'refused'; status: number }
  | { kind: 'uncounted'; status: number }
  | { kind: 'unreached' };

export class Router {
  readonly #envUrl: string;
  readonly #brief: Brief;
  readonly #probes = new Map<string, Probe>();
  readonly #counted = new Map<string, number>();
  readonly #refusalsInRow = new Map<string, number>();
  readonly #refusalsTotal = new Map<string, number>();

  constructor(envUrl: string, brief: Brief, probes: readonly Probe[]) {
    this.#envUrl = envUrl;
    this.#brief = brief;
    for (const probe of probes) this.#probes.set(probe.provider.id, probe);
  }

  /** Whether a provider is demoted to last resort: 3 refusals in all, or its budget range spent. */
  #demoted(providerId: string): boolean {
    if ((this.#refusalsTotal.get(providerId) ?? 0) >= 3) return true;
    if ((this.#refusalsInRow.get(providerId) ?? 0) >= 3) return true;
    return (this.#counted.get(providerId) ?? 0) >= budgetTop(providerId);
  }

  /** Every free model that may serve this request, in the order they should be tried. */
  candidates(view: ClientView, requires: Requirement): Candidate[] {
    const out: Candidate[] = [];
    for (const row of this.#brief.ratings) {
      if (!row.free) continue;
      if (!meetsRequirements(row, requires)) continue;
      const probe = this.#probes.get(row.provider);
      if (probe === undefined || !probe.ok) continue;
      if (!probe.models.some((m) => m.id === row.model)) continue;
      // A JSON-mode client request dies on the anthropic wire's echo: never route it there.
      if (view.jsonMode && probe.provider.wire === 'anthropic') continue;
      out.push({ provider: probe.provider, model: row.model, rating: row.rating_milli });
    }
    const sameWire = (c: Candidate): number => (c.provider.wire === view.api ? 1 : 0);
    out.sort(
      (a, b) =>
        b.rating - a.rating ||
        budgetMid(b.provider.id) - budgetMid(a.provider.id) ||
        sameWire(b) - sameWire(a) ||
        compareStrings(a.provider.id, b.provider.id) ||
        compareStrings(a.model, b.model),
    );
    return out;
  }

  /** One call to one candidate: the answer, a counted refusal, an uncounted refusal, or no answer. */
  async #attempt(candidate: Candidate, view: ClientView): Promise<Attempt> {
    const provider = candidate.provider;
    const base = joinBase(this.#envUrl, provider.base);
    const body = buildUpstream(view, provider.wire, candidate.model);
    const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(provider.auth, provider.key) };
    if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
    let url: string;
    if (provider.wire === 'openai') url = `${base}/v1/chat/completions`;
    else if (provider.wire === 'anthropic') url = `${base}/v1/messages`;
    else url = `${base}/v1beta/models/${encodeURIComponent(candidate.model)}:${view.stream ? 'streamGenerateContent?alt=sse' : 'generateContent'}`;
    if (view.stream && provider.wire !== 'gemini') headers.accept = 'text/event-stream';

    // A concurrency refusal spends no budget and means only that a call is still in flight; the
    // entry calls serially, so it should never arrive — if it does, wait for the slot and retry.
    for (let tries = 0; tries < 4; tries++) {
      let res: Response;
      try {
        res = await fetch(url, { method: 'POST', headers, body: JSON.stringify(body), signal: AbortSignal.timeout(15000) });
      } catch {
        return { kind: 'unreached' };
      }
      const text = await res.text();
      if (res.status >= 200 && res.status < 300) {
        return { kind: 'answer', answer: parseAnswer(provider.wire, view.stream, text) };
      }
      const code = errorCodeOf(text);
      if (res.status === 429 && code === 'concurrent_requests' && tries < 3) {
        await new Promise((resolve) => setTimeout(resolve, 40));
        continue;
      }
      if (res.status === 402 || res.status === 429 || res.status >= 500) return { kind: 'refused', status: res.status };
      return { kind: 'uncounted', status: res.status };
    }
    return { kind: 'uncounted', status: 429 };
  }

  /**
   * Serve one client request: walk the candidates until one answers. Returns the normalized answer
   * and the model that served it, or null when every candidate refused.
   */
  async serve(view: ClientView, requires: Requirement): Promise<{ answer: Normalized; model: string } | null> {
    const candidates = this.candidates(view, requires);
    const live = candidates.filter((c) => !this.#demoted(c.provider.id));
    const ordered = live.length > 0 ? [...live, ...candidates.filter((c) => this.#demoted(c.provider.id))] : candidates;
    const refusedHere = new Map<string, number>();
    for (const candidate of ordered) {
      if ((refusedHere.get(candidate.provider.id) ?? 0) >= 2) continue;
      const attempt = await this.#attempt(candidate, view);
      if (attempt.kind === 'answer') {
        this.#counted.set(candidate.provider.id, (this.#counted.get(candidate.provider.id) ?? 0) + 1);
        this.#refusalsInRow.set(candidate.provider.id, 0);
        return { answer: attempt.answer, model: candidate.model };
      }
      if (attempt.kind === 'refused') {
        this.#counted.set(candidate.provider.id, (this.#counted.get(candidate.provider.id) ?? 0) + 1);
        this.#refusalsInRow.set(candidate.provider.id, (this.#refusalsInRow.get(candidate.provider.id) ?? 0) + 1);
        this.#refusalsTotal.set(candidate.provider.id, (this.#refusalsTotal.get(candidate.provider.id) ?? 0) + 1);
        refusedHere.set(candidate.provider.id, (refusedHere.get(candidate.provider.id) ?? 0) + 1);
      }
      // 'uncounted' (a 400-class answer: our own mistake, no budget spent) and 'unreached' simply
      // move the request to the next candidate.
    }
    return null;
  }
}

/** The failure code in a provider's error envelope, whichever wire spelled it. */
function errorCodeOf(text: string): string | null {
  try {
    const value: unknown = JSON.parse(text);
    if (value === null || typeof value !== 'object' || Array.isArray(value)) return null;
    const error = (value as Json).error;
    if (error === null || typeof error !== 'object' || Array.isArray(error)) return null;
    const e = error as Json;
    if (typeof e.code === 'string') return e.code;
    if (typeof e.status === 'string') return e.status;
    if (typeof e.type === 'string') return e.type;
    return null;
  } catch {
    return null;
  }
}
