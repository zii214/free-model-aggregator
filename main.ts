// free-model-aggregator-01: one server, two client wires, three provider wires, routed by rating.
//
// The trial gives this server a brief (the provider directory, the sealed quality table's public copy,
// and the workload manifest) and an env that hosts the ten simulated providers. The server probes each
// provider's catalog in its own wire, registers one catalog, then answers the client's twenty requests
// one at a time. For each request it walks the free models that meet the request's requirements in
// rating order, speaks the chosen provider's wire, and fails over to the next candidate whenever a
// provider refuses (402, 429 or 5xx) — a refusal spends some of that provider's per-trial budget but is
// never the answer returned. Answers are normalized and re-rendered in the client's wire (JSON or SSE),
// with the provider's usage and stop reason carried across.
//
// Provider budgets are drawn per trial and never published, so the router tracks what it has seen:
// every counted call (a success or a refusal) per provider, and how many refusals came in a row. The
// generator's failure schedules refuse at most two isolated calls per provider, so three refusals in a
// row — or reaching the top of the provider's published budget range — means the provider is spent or
// out for the rest of the trial: it drops behind every live candidate and is only retried when nothing
// live can serve the request. Every choice is a function of the brief and of the answers seen so far:
// no wall clock, no randomness, so a replay reproduces the trial byte for byte.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  budgetMax,
  budgetMid,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type ProviderDirectoryEntry,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicModelList,
  buildUpstream,
  echoFromAnthropic,
  echoFromOpenAI,
  openAIError,
  openAIModelList,
  parseAnswer,
  parseClient,
  renderAnswer,
  type ClientWire,
  type EchoFields,
  type NormAnswer,
  type NormRequest,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one provider call may take before the router fails over to the next candidate. */
const CALL_TIMEOUT_MS = 25000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** Refusals in a row after which a provider is treated as spent or out for the rest of the trial. */
const DEAD_AFTER_REFUSALS = 3;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

/** The free models this entry serves, one row per model id: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  const seen = new Set<string>();
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Routing: candidates in rating order, with per-provider state learned from
// the refusals and successes the trial has shown so far.

interface ProviderState {
  /** Counted calls made to this provider: successes and refusals alike spend its budget. */
  counted: number;
  /** Refusals since the last success; three in a row can only be a spent budget or an outage. */
  consecutiveRefusals: number;
  /** Whether this provider's most recent call refused. */
  lastRefused: boolean;
  /** Certainly spent or out: retried only when no live candidate can serve a request. */
  dead: boolean;
}

const providerState = new Map<string, ProviderState>();

function stateOf(providerId: string): ProviderState {
  let state = providerState.get(providerId);
  if (state === undefined) {
    state = { counted: 0, consecutiveRefusals: 0, lastRefused: false, dead: false };
    providerState.set(providerId, state);
  }
  return state;
}

interface Candidate {
  provider: ProviderDirectoryEntry;
  model: ProbedModel;
  rating: number;
}

/**
 * Every free model that may serve one request, best first: rating, then the roomier provider's budget,
 * then the provider whose wire is the client's (a pass-through carries no translation risk), then ids
 * for a stable order. A request in OpenAI JSON mode also drops the Anthropic wire here: it cannot carry
 * response_format, so its echo would report the mode absent and lose the request.
 */
function candidatesFor(requires: Requirement, client: ClientWire, jsonCarry: boolean): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (jsonCarry && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ provider: probe.provider, model, rating: row?.rating_milli ?? 0 });
    }
  }
  out.sort((a, b) => {
    if (a.rating !== b.rating) return b.rating - a.rating;
    const mid = budgetMid(b.provider.id) - budgetMid(a.provider.id);
    if (mid !== 0) return mid;
    const wireA = a.provider.wire === client ? 0 : 1;
    const wireB = b.provider.wire === client ? 0 : 1;
    if (wireA !== wireB) return wireA - wireB;
    const refused = (stateOf(a.provider.id).lastRefused ? 1 : 0) - (stateOf(b.provider.id).lastRefused ? 1 : 0);
    if (refused !== 0) return refused;
    if (a.provider.id !== b.provider.id) return a.provider.id < b.provider.id ? -1 : 1;
    return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
  });
  return out;
}

/** Live candidates in order, then the certainly-spent ones: a dead provider is a last resort, not a skip. */
function orderedCandidates(requires: Requirement, client: ClientWire, jsonCarry: boolean): Candidate[] {
  const all = candidatesFor(requires, client, jsonCarry);
  const live = all.filter((c) => !stateOf(c.provider.id).dead);
  const dead = all.filter((c) => stateOf(c.provider.id).dead);
  return [...live, ...dead];
}

function noteSuccess(providerId: string): void {
  const state = stateOf(providerId);
  state.counted += 1;
  state.consecutiveRefusals = 0;
  state.lastRefused = false;
}

function noteRefusal(providerId: string, counted: boolean): void {
  const state = stateOf(providerId);
  if (counted) state.counted += 1;
  state.consecutiveRefusals += 1;
  state.lastRefused = true;
  if (state.consecutiveRefusals >= DEAD_AFTER_REFUSALS || state.counted >= budgetMax(providerId)) state.dead = true;
}

// ---------------------------------------------------------------------------
// One provider call, in the provider's own wire.

interface Attempt {
  ok: boolean;
  /** Whether the call reached the provider and named a free model (so it spent budget when refused). */
  counted: boolean;
  answer: NormAnswer | null;
  model: string;
}

async function attempt(candidate: Candidate, norm: NormRequest, clientEcho: EchoFields, clientHeaders: IncomingMessage['headers']): Promise<Attempt> {
  const provider = candidate.provider;
  const model = candidate.model.id;
  const body = buildUpstream(provider.wire, norm, model, clientEcho);
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  let url: string;
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    if (norm.stream) headers.accept = 'text/event-stream';
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    const version = clientHeaders['anthropic-version'];
    headers['anthropic-version'] = typeof version === 'string' && version !== '' ? version : '2023-06-01';
    if (norm.stream) headers.accept = 'text/event-stream';
  } else {
    url = `${base}/v1beta/models/${encodeURIComponent(model)}:${norm.stream ? 'streamGenerateContent' : 'generateContent'}`;
    if (norm.stream) headers.accept = 'text/event-stream';
  }
  let res: Response;
  try {
    res = await fetch(url, { method: 'POST', headers, body: JSON.stringify(body), signal: AbortSignal.timeout(CALL_TIMEOUT_MS) });
  } catch {
    return { ok: false, counted: false, answer: null, model };
  }
  const text = await res.text();
  if (res.status >= 200 && res.status < 300) {
    return { ok: true, counted: true, answer: parseAnswer(provider.wire, text, norm.stream), model };
  }
  const counted = res.status === 402 || res.status === 429 || res.status >= 500;
  return { ok: false, counted, answer: null, model };
}

// ---------------------------------------------------------------------------
// The client routes.

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(norm: NormRequest, body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(norm, body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(norm: NormRequest, _body: Json): Requirement {
  let vision = false;
  for (const message of norm.messages) {
    for (const part of message.parts) if (part.kind === 'image' || part.kind === 'imageUrl') vision = true;
  }
  return {
    tools: norm.tools.length > 0,
    vision,
    json: norm.jsonMode,
  };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: ClientWire, status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** Serve one client request on either wire: route by rating, fail over on refusals, render the answer. */
async function serve(req: IncomingMessage, res: ServerResponse, client: ClientWire): Promise<void> {
  const body = await readJson(req);
  if (body === null) {
    return sendError(res, client, 400, 'invalid_request_error', 'the request body is not one JSON object');
  }
  const norm = parseClient(client, body);
  const clientEcho = client === 'openai' ? echoFromOpenAI(body) : echoFromAnthropic(body);
  const requires = requirementsFor(norm, body);
  const jsonCarry = client === 'openai' && norm.jsonMode;
  const candidates = orderedCandidates(requires, client, jsonCarry);
  const tried = new Set<string>();
  for (const candidate of candidates) {
    if (tried.has(candidate.provider.id)) continue;
    tried.add(candidate.provider.id);
    const result = await attempt(candidate, norm, clientEcho, req.headers);
    if (result.ok && result.answer !== null) {
      noteSuccess(candidate.provider.id);
      const rendered = renderAnswer(client, result.answer, result.model, norm.stream);
      res.writeHead(200, { 'content-type': rendered.contentType });
      res.end(rendered.body);
      return;
    }
    noteRefusal(candidate.provider.id, result.counted);
  }
  sendError(res, client, 502, client === 'anthropic' ? 'api_error' : 'upstream_error', 'no free model on the directory could serve this request');
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await serve(req, res, 'openai');
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await serve(req, res, 'anthropic');
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/**
 * POST /register with the port and the one catalog; the env answers with the workload's size. The env is
 * up before the trial starts, but a refused first connection is retried for a few seconds rather than
 * lost: the register body is the same document on every attempt, and the env keeps the first catalog it
 * is given, so a retry can never change what is registered.
 */
async function register(port: number, report: CatalogReport): Promise<void> {
  const body = JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report });
  let lastError: unknown = null;
  for (let attemptNr = 0; attemptNr < 20; attemptNr++) {
    try {
      const res = await fetch(`${ENV_URL}/register`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body,
        signal: AbortSignal.timeout(5000),
      });
      if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
      const reply: unknown = await res.json();
      if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
      return;
    } catch (err) {
      lastError = err;
      await new Promise((resolve) => setTimeout(resolve, 250));
    }
  }
  throw lastError instanceof Error ? lastError : new Error('register failed');
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
