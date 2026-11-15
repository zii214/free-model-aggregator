// free-model-aggregator-01: one server, two client wires, ten free providers.
//
// The server reads the brief at $MAKEFUN_PROMPT_FILE, probes every provider's catalog in its own
// wire, builds the one catalog, listens on a port of its own, and registers {protocol_version: 1,
// port, catalog} with the env at $MAKEFUN_ENV_URL. The env's client then sends twenty requests, one
// at a time — POST /v1/chat/completions on the OpenAI wire, POST /v1/messages on the Anthropic one,
// each as JSON or as server-sent events — and each is routed to the best free model that meets its
// requirements and still has budget (lib/router.ts), translated across wires under an echo guard
// (lib/wire.ts), and answered in the client's wire with the provider's text whole, its usage counts
// field-mapped and its stop reason family-mapped. GET /v1/models answers in the shape of the auth
// header it was asked with. The server polls GET /work and exits 0 the moment the env says done.
//
// `node main.ts --list` prints the catalog document, byte for byte the one the register body
// carries, and exits 0 before any server starts; with no brief to read it prints {"providers":[]}.
//
// Nothing here reads a clock or a random source for a decision: every route is a function of the
// brief, the requests, and the providers' answers seen so far, so a replayed trial reproduces the
// env's state export byte for byte.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  isRecord,
  probeProvider,
  readBrief,
  type Brief,
  type Catalog,
  type Json,
  type Probe,
  type Requirement,
} from './lib/brief.ts';
import { Router } from './lib/router.ts';
import { anthropicModelList, errorBody, openAIModelList, renderAnswer, viewOf, type ServedModel } from './lib/wire.ts';

const PROTOCOL_VERSION = 1;
const POLL_INTERVAL_MS = 100;
const PROBE_TIMEOUT_MS = 5000;
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

interface Loaded {
  brief: Brief;
  probes: Probe[];
  catalog: Catalog;
}

/** Read the brief, probe every provider in directory order, and build the one catalog. */
async function load(): Promise<Loaded> {
  const brief = readBrief(PROMPT_FILE);
  if (brief.error !== null) log(brief.error);
  const probes = await Promise.all(brief.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  return { brief, probes, catalog: buildCatalog(brief, probes) };
}

/** The free model ids the registered catalog holds, each once, in catalog order. */
function servedModels(catalog: Catalog): ServedModel[] {
  const seen = new Set<string>();
  const out: ServedModel[] = [];
  for (const provider of catalog.providers) {
    for (const model of provider.models) {
      if (!model.free || seen.has(model.id)) continue;
      seen.add(model.id);
      out.push({ id: model.id, provider: provider.id });
    }
  }
  return out;
}

/** Requirements read off a request body, for a brief whose manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
    json: format === 'json_object' || format === 'json_schema',
  };
}

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

function send(res: ServerResponse, status: number, contentType: string, body: string): void {
  res.writeHead(status, { 'content-type': contentType, 'content-length': Buffer.byteLength(body) });
  res.end(body);
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  send(res, status, 'application/json', JSON.stringify(body));
}

class App {
  readonly #loaded: Loaded;
  readonly #router: Router;
  readonly #models: ServedModel[];
  /** Client API requests answered so far: the manifest is in arrival order. */
  #arrived = 0;

  constructor(loaded: Loaded) {
    this.#loaded = loaded;
    this.#router = new Router(ENV_URL, loaded.brief, loaded.probes);
    this.#models = servedModels(loaded.catalog);
  }

  async handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
    const url = new URL(req.url ?? '/', 'http://entry');
    if (req.method === 'GET' && url.pathname === '/v1/models') {
      if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(this.#models));
      else sendJson(res, 200, openAIModelList(this.#models));
      return;
    }
    if (req.method === 'POST' && (url.pathname === '/v1/chat/completions' || url.pathname === '/v1/messages')) {
      await this.#serve(url.pathname === '/v1/messages' ? 'anthropic' : 'openai', req, res);
      return;
    }
    const api = url.pathname === '/v1/messages' ? 'anthropic' : 'openai';
    sendJson(res, 404, errorBody(api, api === 'anthropic' ? 'not_found_error' : 'not_found', `no route ${req.method ?? ''} ${url.pathname}`));
  }

  async #serve(api: 'openai' | 'anthropic', req: IncomingMessage, res: ServerResponse): Promise<void> {
    const body = await readJson(req);
    if (body === null) {
      sendJson(res, 400, errorBody(api, 'invalid_request_error', 'the request body is not one JSON object'));
      return;
    }
    const index = this.#arrived;
    this.#arrived += 1;
    const manifest = this.#loaded.brief.manifest;
    const requires = index < manifest.length ? manifest[index].requires : requirementsFromBody(body);
    const view = viewOf(api, body);
    const served = await this.#router.serve(view, requires);
    if (served === null) {
      sendJson(res, 503, errorBody(api, api === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model on the directory could serve this request'));
      return;
    }
    const rendered = renderAnswer(api, served.model, served.answer, view.stream);
    send(res, rendered.status, rendered.contentType, rendered.body);
  }
}

function listen(server: Server): Promise<number> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => {
      const address = server.address();
      if (address === null || typeof address === 'string') reject(new Error('the server has no port'));
      else resolve(address.port);
    });
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, catalog: Catalog): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
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
        if (res.status === 200) {
          const work: unknown = await res.json();
          if (isRecord(work) && work.done === true) {
            clearInterval(timer);
            server.close();
            process.exit(0);
          }
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
  const loaded = await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and
    // stop before any server starts. Wait for stdout to flush so the tail is never lost.
    const text = `${JSON.stringify(loaded.catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const app = new App(loaded);
  const server = createServer((req, res) => {
    void app.handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) {
        sendJson(res, 500, errorBody(req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 'internal_error', 'the server failed to answer'));
      }
    });
  });
  const port = await listen(server);
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${loaded.catalog.providers.length} answered providers and ${servedModels(loaded.catalog).length} free models`);
  await register(port, loaded.catalog);
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
