// The brief: the three JSON documents a trial hands the entry, found by their `schema` member in the
// prose at $MAKEFUN_PROMPT_FILE, plus the catalog probe every provider answers in its own wire.
//
//   - providers/v1: the directory. One row a provider: id, name, wire (openai | anthropic | gemini),
//     auth scheme, the key variable a real deployment reads, the key this simulated service takes, and
//     the base path (`/p/<id>`) the entry joins to $MAKEFUN_ENV_URL to dial it.
//   - ratings/v1: the quality table. One row a FREE model (paid models are not in the brief; the probe
//     below is how the entry learns of them): rating_milli, context_length, and the tools / vision /
//     json capabilities the scorer holds it to.
//   - workload/v1: the manifest. The twenty requests in arrival order: api, stream, weight, requires.
//
// The probe asks each provider's own catalog route (openai and anthropic at /v1/models, gemini at
// /v1beta/models) with the provider's own auth header, and the catalog the entry registers is built
// from what the probes answered: the providers' own listings are the truth the catalog is scored
// against, with the quality table's free flag standing where a row exists.
//
// Everything in this file is a pure function of the brief and the probe answers: no clock, no
// randomness, so the document `--list` prints and the one the register body carries are one object.

import { readFileSync } from 'node:fs';

export type Wire = 'openai' | 'anthropic' | 'gemini';
export type AuthScheme = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';

export const PROVIDERS_SCHEMA = 'free-model-aggregator-01/providers/v1';
export const RATINGS_SCHEMA = 'free-model-aggregator-01/ratings/v1';
export const WORKLOAD_SCHEMA = 'free-model-aggregator-01/workload/v1';

export type Json = Record<string, unknown>;

export function isRecord(v: unknown): v is Json {
  return v !== null && typeof v === 'object' && !Array.isArray(v);
}

/** One provider as the directory lists it. `base` is its path (or an absolute base URL). */
export interface ProviderEntry {
  id: string;
  name: string;
  wire: Wire;
  auth: AuthScheme;
  base: string;
  key_env: string;
  key: string;
}

/** One row of the quality table: a free model, its rating and what it can do. */
export interface RatingRow {
  model: string;
  provider: string;
  rating_milli: number;
  context_length: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

/** What one request needs from the model that serves it. */
export interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

export interface WorkloadRequest {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: 1 | 2;
  requires: Requirement;
}

export interface Brief {
  providers: ProviderEntry[];
  ratings: RatingRow[];
  manifest: WorkloadRequest[];
  error: string | null;
}

/** One model as a provider's catalog route listed it; a null member is one the listing did not say. */
export interface ListedModel {
  id: string;
  free: boolean | null;
  context_length: number | null;
}

/** The answer one provider's catalog route gave, in directory order beside the others. */
export interface Probe {
  provider: ProviderEntry;
  ok: boolean;
  models: ListedModel[];
}

export interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

export interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: Wire;
  models: CatalogModel[];
}

/** The catalog document: what `--list` prints and what the register body carries, byte for byte. */
export interface Catalog {
  providers: CatalogProvider[];
}

function asString(v: unknown): string {
  return typeof v === 'string' ? v : '';
}

function asFiniteNumber(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

/** The closing brace of the object that opens at `start`, or -1; string contents are skipped. */
function objectEnd(text: string, start: number): number {
  let depth = 0;
  let quoted = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (quoted) {
      if (ch === '\\') i += 1;
      else if (ch === '"') quoted = false;
    } else if (ch === '"') quoted = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

/** Every JSON object embedded in `text` that carries a string `schema`, in the order they appear. */
function embeddedDocuments(text: string): Json[] {
  const found: Json[] = [];
  for (let at = text.indexOf('{'); at !== -1; at = text.indexOf('{', at + 1)) {
    const end = objectEnd(text, at);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(at, end + 1));
      if (isRecord(value) && typeof value.schema === 'string') found.push(value);
    } catch {
      // Not a document on its own: keep looking.
    }
  }
  return found;
}

function providerFrom(v: unknown): ProviderEntry | null {
  if (!isRecord(v)) return null;
  const id = asString(v.id);
  const wire = v.wire === 'openai' || v.wire === 'anthropic' || v.wire === 'gemini' ? v.wire : null;
  if (id === '' || wire === null) return null;
  const auth =
    v.auth === 'bearer' || v.auth === 'x-api-key' || v.auth === 'x-goog-api-key' || v.auth === 'none' ? v.auth : 'bearer';
  // The directory writes `path`; a directory spelling an absolute base may use one of the others.
  const base = [v.path, v.base_path, v.base_url, v.base].map(asString).find((s) => s !== '') ?? '';
  return { id, name: asString(v.name) !== '' ? asString(v.name) : id, wire, auth, base, key_env: asString(v.key_env), key: asString(v.key) };
}

function ratingFrom(v: unknown): RatingRow | null {
  if (!isRecord(v)) return null;
  const model = asString(v.model);
  const provider = asString(v.provider);
  if (model === '' || provider === '') return null;
  return {
    model,
    provider,
    rating_milli: asFiniteNumber(v.rating_milli) ?? 0,
    context_length: asFiniteNumber(v.context_length),
    tools: v.tools === true,
    vision: v.vision === true,
    json: v.json === true,
    free: v.free === true,
  };
}

function requestFrom(v: unknown): WorkloadRequest | null {
  if (!isRecord(v)) return null;
  const id = asString(v.id);
  if (id === '') return null;
  const requires = isRecord(v.requires) ? v.requires : {};
  return {
    id,
    api: v.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: v.stream === true,
    weight: v.weight === 2 ? 2 : 1,
    requires: {
      tools: requires.tools === true,
      vision: requires.vision === true,
      json: requires.json === true,
      ...(asFiniteNumber(requires.min_context) !== null ? { min_context: asFiniteNumber(requires.min_context) as number } : {}),
    },
  };
}

/** The brief at `path`: its directory, quality table and manifest, each found by schema. */
export function readBrief(path: string): Brief {
  const empty: Brief = { providers: [], ratings: [], manifest: [], error: null };
  if (path === '') return { ...empty, error: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { ...empty, error: `cannot read the brief at ${path}: ${(err as Error).message}` };
  }
  const docs = embeddedDocuments(text);
  const directory = docs.find((d) => d.schema === PROVIDERS_SCHEMA && Array.isArray(d.providers));
  if (directory === undefined) return { ...empty, error: `the brief at ${path} holds no ${PROVIDERS_SCHEMA} document` };
  const providers = (directory.providers as unknown[]).flatMap((v) => {
    const p = providerFrom(v);
    return p === null ? [] : [p];
  });
  const ratingsDoc = docs.find((d) => d.schema === RATINGS_SCHEMA && Array.isArray(d.ratings));
  const ratings =
    ratingsDoc === undefined
      ? []
      : (ratingsDoc.ratings as unknown[]).flatMap((v) => {
          const r = ratingFrom(v);
          return r === null ? [] : [r];
        });
  const workloadDoc = docs.find((d) => d.schema === WORKLOAD_SCHEMA && Array.isArray(d.manifest));
  const manifest =
    workloadDoc === undefined
      ? []
      : (workloadDoc.manifest as unknown[]).flatMap((v) => {
          const r = requestFrom(v);
          return r === null ? [] : [r];
        });
  return { providers, ratings, manifest, error: null };
}

/** The quality table's row for one provider's model, when it has one. */
export function ratingRow(brief: Brief, providerId: string, modelId: string): RatingRow | null {
  return brief.ratings.find((r) => r.provider === providerId && r.model === modelId) ?? null;
}

/** A provider's base path joined to the env's origin; an absolute base stands as it is. */
export function joinBase(envUrl: string, base: string): string {
  const b = base.trim();
  if (/^https?:\/\//i.test(b)) return b.replace(/\/+$/, '');
  const origin = envUrl.replace(/\/+$/, '');
  if (b === '') return origin;
  return `${origin}${b.startsWith('/') ? '' : '/'}${b.replace(/\/+$/, '')}`;
}

/** The header a provider's auth scheme carries its key in; a keyless scheme sends none. */
export function authHeaders(auth: AuthScheme, key: string): Record<string, string> {
  if (key === '') return {};
  if (auth === 'bearer') return { authorization: `Bearer ${key}` };
  if (auth === 'x-api-key') return { 'x-api-key': key };
  if (auth === 'x-goog-api-key') return { 'x-goog-api-key': key };
  return {};
}

/** A listing's word on price: free when every price it names is zero, null when it names none. */
function listingFree(v: Json): boolean | null {
  if (typeof v.free === 'boolean') return v.free;
  const pricing = isRecord(v.pricing) ? v.pricing : null;
  if (pricing === null) return null;
  const amounts = Object.values(pricing).flatMap((x) => {
    if (typeof x === 'number') return [x];
    if (typeof x === 'string' && x.trim() !== '' && Number.isFinite(Number(x))) return [Number(x)];
    return [];
  });
  if (amounts.length === 0) return null;
  return amounts.every((n) => n === 0);
}

function listedModel(wire: Wire, v: unknown): ListedModel | null {
  if (!isRecord(v)) return null;
  const id = wire === 'gemini' ? asString(v.name).replace(/^models\//, '') : asString(v.id);
  if (id === '') return null;
  return {
    id,
    free: listingFree(v),
    context_length: asFiniteNumber(v.context_length) ?? asFiniteNumber(v.context_window) ?? asFiniteNumber(v.inputTokenLimit),
  };
}

/** Probe one provider's catalog route in its own wire; every failure leaves the provider unprobed. */
export async function probeProvider(envUrl: string, provider: ProviderEntry, timeoutMs = 5000): Promise<Probe> {
  const base = joinBase(envUrl, provider.base);
  const url = provider.wire === 'gemini' ? `${base}/v1beta/models` : `${base}/v1/models`;
  const headers: Record<string, string> = { accept: 'application/json', ...authHeaders(provider.auth, provider.key) };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  try {
    const res = await fetch(url, { headers, signal: AbortSignal.timeout(timeoutMs) });
    if (!res.ok) return { provider, ok: false, models: [] };
    const body: unknown = await res.json();
    if (!isRecord(body)) return { provider, ok: false, models: [] };
    const list = provider.wire === 'gemini' ? body.models : body.data;
    if (!Array.isArray(list)) return { provider, ok: false, models: [] };
    return {
      provider,
      ok: true,
      models: list.flatMap((v) => {
        const m = listedModel(provider.wire, v);
        return m === null ? [] : [m];
      }),
    };
  } catch {
    return { provider, ok: false, models: [] };
  }
}

/**
 * The one catalog, from the probes and the quality table: every provider that answered, every model
 * it listed, models sorted by id. A model's free flag is the quality table's where a row exists (the
 * sealed truth) and the listing's own word where none does; a model neither vouches for is paid.
 * The base_url is the directory's own path, never joined to the env's origin: the origin differs
 * between a trial and its replay, and the catalog is part of what a replay must reproduce.
 */
export function buildCatalog(brief: Brief, probes: readonly Probe[]): Catalog {
  const providers: CatalogProvider[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    const seen = new Set<string>();
    const models: CatalogModel[] = [];
    const sorted = [...probe.models].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
    for (const model of sorted) {
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      const row = ratingRow(brief, probe.provider.id, model.id);
      models.push({
        id: model.id,
        free: row !== null ? row.free : (model.free ?? false),
        context_length: row?.context_length ?? model.context_length ?? null,
      });
    }
    providers.push({ id: probe.provider.id, base_url: probe.provider.base, key_env: probe.provider.key_env, wire: probe.provider.wire, models });
  }
  return { providers };
}
