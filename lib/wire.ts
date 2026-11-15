// The wires: how a client request becomes a provider request, and how a provider's answer becomes
// the client's answer.
//
// A request is served in three moves. First the client body is read once into a ClientView (its own
// wire's parse, the last image, its tools). Then an upstream body is built for the provider's wire —
// a faithful translation of the client's body — and VERIFIED: the provider wire's parser must make
// of it the very echo the client's parser made of the client body. A translation that fails the
// check (a conversation ending in a tool result is the classic case: one wire's last word is another
// wire's silence) is replaced by a canonical body built from the ClientView alone, which parses to
// the same echo by construction. Only then is the call made.
//
// A provider's answer is never piped through: it is parsed into one normalized shape — the whole
// text, the usage counts, the stop reason — and re-rendered in the client's wire, JSON or SSE, with
// the usage field-mapped and the stop reason mapped by family (a provider's finish lands in the
// client's token of the same family: end_turn to stop, max_tokens to length, tool_use to tool_calls,
// refusal to content_filter, and back).

import { isRecord, type Json, type Wire } from './brief.ts';
import { anthropicParsed, clientEcho, echoEquals, openAIParsed, upstreamEcho, type Echo } from './echo.ts';

/** One tool the client declared, in wire-neutral shape. */
export interface ClientTool {
  name: string;
  description: string;
  parameters: Json;
}

/** The last image a client sent: its bytes inline, or a URL they live at. */
export type ClientImage = { kind: 'data'; mediaType: string; data: string } | { kind: 'url'; url: string };

/** Everything about one client request that an upstream body or an answer needs. */
export interface ClientView {
  api: 'openai' | 'anthropic';
  stream: boolean;
  echo: Echo;
  systemText: string;
  userText: string;
  image: ClientImage | null;
  tools: ClientTool[];
  maxTokens: number | null;
  stops: string[];
  jsonMode: boolean;
  temperature: number | null;
  topP: number | null;
  raw: Json;
}

/** A provider's answer, normalized: the whole text, its usage, its stop reason in its own wire. */
export interface Normalized {
  text: string;
  input: number | null;
  output: number | null;
  stop: string | null;
}

function numberOrNull(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

function textOfBlocks(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('\n');
}

function imageFromUrl(url: string): ClientImage {
  const data = /^data:([^,]*);base64,(.*)$/s.exec(url);
  if (data !== null) return { kind: 'data', mediaType: data[1] === '' ? 'image/png' : data[1], data: data[2] };
  return { kind: 'url', url };
}

/** Read one client body into its view: the parse, the last image, the tools, the caps. */
export function viewOf(api: 'openai' | 'anthropic', raw: Json): ClientView {
  const parsed = api === 'openai' ? openAIParsed(raw) : anthropicParsed(raw);
  let image: ClientImage | null = null;
  const tools: ClientTool[] = [];
  const messages = Array.isArray(raw.messages) ? raw.messages.filter(isRecord) : [];
  if (api === 'openai') {
    for (const message of messages) {
      if (message.role !== 'user' || !Array.isArray(message.content)) continue;
      for (const block of message.content) {
        if (isRecord(block) && block.type === 'image_url' && isRecord(block.image_url) && typeof block.image_url.url === 'string') {
          image = imageFromUrl(block.image_url.url);
        }
      }
    }
    if (Array.isArray(raw.tools)) {
      for (const tool of raw.tools) {
        if (!isRecord(tool) || !isRecord(tool.function) || typeof tool.function.name !== 'string') continue;
        tools.push({
          name: tool.function.name,
          description: typeof tool.function.description === 'string' ? tool.function.description : '',
          parameters: isRecord(tool.function.parameters) ? tool.function.parameters : { type: 'object', properties: {} },
        });
      }
    }
  } else {
    for (const message of messages) {
      if (message.role !== 'user' || !Array.isArray(message.content)) continue;
      for (const block of message.content) {
        if (!isRecord(block) || block.type !== 'image' || !isRecord(block.source)) continue;
        if (block.source.type === 'base64' && typeof block.source.data === 'string') {
          image = { kind: 'data', mediaType: typeof block.source.media_type === 'string' ? block.source.media_type : 'image/png', data: block.source.data };
        } else if (block.source.type === 'url' && typeof block.source.url === 'string') {
          image = { kind: 'url', url: block.source.url };
        }
      }
    }
    if (Array.isArray(raw.tools)) {
      for (const tool of raw.tools) {
        if (!isRecord(tool) || typeof tool.name !== 'string') continue;
        tools.push({
          name: tool.name,
          description: typeof tool.description === 'string' ? tool.description : '',
          parameters: isRecord(tool.input_schema) ? tool.input_schema : { type: 'object', properties: {} },
        });
      }
    }
  }
  return {
    api,
    stream: raw.stream === true,
    echo: clientEcho(api, raw),
    systemText: parsed.systemText,
    userText: parsed.userText,
    image,
    tools,
    maxTokens: parsed.echo.max_tokens,
    stops: parsed.echo.stops,
    jsonMode: parsed.echo.json_mode,
    temperature: numberOrNull(raw.temperature),
    topP: numberOrNull(raw.top_p),
    raw,
  };
}

// ---------------------------------------------------------------------------
// Faithful translations: the client's body rewritten in the provider's wire.
// ---------------------------------------------------------------------------

function argumentsObject(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOfBlocks(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** A tool-call id's function name, from any earlier assistant turn of the conversation. */
function toolNameById(raw: Json, id: string): string {
  const messages = Array.isArray(raw.messages) ? raw.messages.filter(isRecord) : [];
  for (const message of messages) {
    if (Array.isArray(message.tool_calls)) {
      for (const call of message.tool_calls) {
        if (isRecord(call) && call.id === id && isRecord(call.function) && typeof call.function.name === 'string') return call.function.name;
      }
    }
    if (Array.isArray(message.content)) {
      for (const block of message.content) {
        if (isRecord(block) && block.type === 'tool_use' && block.id === id && typeof block.name === 'string') return block.name;
      }
    }
  }
  return '';
}

function openAIImageBlock(image: ClientImage): Json {
  return image.kind === 'data'
    ? { type: 'image_url', image_url: { url: `data:${image.mediaType};base64,${image.data}` } }
    : { type: 'image_url', image_url: { url: image.url } };
}

function anthropicImageBlock(image: ClientImage): Json {
  return image.kind === 'data'
    ? { type: 'image', source: { type: 'base64', media_type: image.mediaType, data: image.data } }
    : { type: 'image', source: { type: 'url', url: image.url } };
}

function geminiImagePart(image: ClientImage): Json {
  return image.kind === 'data' ? { inlineData: { mimeType: image.mediaType, data: image.data } } : { fileData: { fileUri: image.url } };
}

/** The client's messages as Anthropic messages (the system prompt rides in its own field). */
function messagesToAnthropic(raw: Json): Json[] {
  const out: Json[] = [];
  const messages = Array.isArray(raw.messages) ? raw.messages.filter(isRecord) : [];
  for (const message of messages) {
    if (message.role === 'system') continue;
    if (message.role === 'tool' || message.role === 'function') {
      out.push({
        role: 'user',
        content: [
          {
            type: 'tool_result',
            tool_use_id: typeof message.tool_call_id === 'string' ? message.tool_call_id : '',
            content: toolResultText(message.content),
          },
        ],
      });
      continue;
    }
    const assistant = message.role === 'assistant';
    if (typeof message.content === 'string') {
      out.push({ role: assistant ? 'assistant' : 'user', content: message.content });
      continue;
    }
    if (!Array.isArray(message.content)) continue;
    const blocks: Json[] = [];
    for (const block of message.content) {
      if (!isRecord(block)) continue;
      if (block.type === 'text' && typeof block.text === 'string') blocks.push({ type: 'text', text: block.text });
      else if (block.type === 'image_url' && isRecord(block.image_url) && typeof block.image_url.url === 'string') {
        blocks.push(anthropicImageBlock(imageFromUrl(block.image_url.url)));
      }
    }
    if (assistant && Array.isArray(message.tool_calls)) {
      for (const call of message.tool_calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        blocks.push({
          type: 'tool_use',
          id: typeof call.id === 'string' ? call.id : '',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input: argumentsObject(call.function.arguments),
        });
      }
    }
    out.push({ role: assistant ? 'assistant' : 'user', content: blocks });
  }
  return out;
}

/** The client's messages as OpenAI messages (the system prompt becomes a system message). */
function messagesToOpenAI(view: ClientView): Json[] {
  const raw = view.raw;
  const out: Json[] = [];
  if (view.systemText !== '') out.push({ role: 'system', content: view.systemText });
  const messages = Array.isArray(raw.messages) ? raw.messages.filter(isRecord) : [];
  for (const message of messages) {
    const assistant = message.role === 'assistant';
    if (typeof message.content === 'string') {
      out.push({ role: assistant ? 'assistant' : 'user', content: message.content });
      continue;
    }
    if (!Array.isArray(message.content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of message.content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: { name: typeof block.name === 'string' ? block.name : '', arguments: JSON.stringify(isRecord(block.input) ? block.input : {}) },
        });
      }
      const text = textOfBlocks(message.content);
      out.push({ role: 'assistant', content: text !== '' ? text : null, ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}) });
      continue;
    }
    const parts: Json[] = [];
    for (const block of message.content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        out.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
      } else if (block.type === 'text' && typeof block.text === 'string') {
        parts.push({ type: 'text', text: block.text });
      } else if (block.type === 'image' && isRecord(block.source)) {
        if (block.source.type === 'base64' && typeof block.source.data === 'string') {
          const media = typeof block.source.media_type === 'string' ? block.source.media_type : 'image/png';
          parts.push({ type: 'image_url', image_url: { url: `data:${media};base64,${block.source.data}` } });
        } else if (block.source.type === 'url' && typeof block.source.url === 'string') {
          parts.push({ type: 'image_url', image_url: { url: block.source.url } });
        }
      }
    }
    if (parts.length > 0) out.push({ role: 'user', content: parts });
  }
  return out;
}

/** The client's messages as Gemini contents (assistant turns are `model` turns). */
function messagesToGemini(view: ClientView): Json[] {
  const raw = view.raw;
  const out: Json[] = [];
  const messages = Array.isArray(raw.messages) ? raw.messages.filter(isRecord) : [];
  for (const message of messages) {
    if (message.role === 'system') continue;
    if (message.role === 'tool' || message.role === 'function') {
      const id = typeof message.tool_call_id === 'string' ? message.tool_call_id : '';
      out.push({
        role: 'user',
        parts: [{ functionResponse: { name: toolNameById(raw, id), response: { result: toolResultText(message.content) } } }],
      });
      continue;
    }
    const assistant = message.role === 'assistant';
    const parts: Json[] = [];
    if (typeof message.content === 'string') {
      if (message.content !== '') parts.push({ text: message.content });
    } else if (Array.isArray(message.content)) {
      for (const block of message.content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
        else if (block.type === 'image_url' && isRecord(block.image_url) && typeof block.image_url.url === 'string') {
          parts.push(geminiImagePart(imageFromUrl(block.image_url.url)));
        } else if (block.type === 'image' && isRecord(block.source)) {
          if (block.source.type === 'base64' && typeof block.source.data === 'string') {
            parts.push({ inlineData: { mimeType: typeof block.source.media_type === 'string' ? block.source.media_type : 'image/png', data: block.source.data } });
          } else if (block.source.type === 'url' && typeof block.source.url === 'string') {
            parts.push({ fileData: { fileUri: block.source.url } });
          }
        } else if (block.type === 'tool_use') {
          parts.push({
            functionCall: { name: typeof block.name === 'string' ? block.name : '', args: isRecord(block.input) ? block.input : {} },
          });
        } else if (block.type === 'tool_result') {
          const id = typeof block.tool_use_id === 'string' ? block.tool_use_id : '';
          parts.push({ functionResponse: { name: toolNameById(raw, id), response: { result: toolResultText(block.content) } } });
        }
      }
    }
    if (assistant && Array.isArray(message.tool_calls)) {
      for (const call of message.tool_calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        parts.push({
          functionCall: {
            name: typeof call.function.name === 'string' ? call.function.name : '',
            args: argumentsObject(call.function.arguments),
          },
        });
      }
    }
    out.push({ role: assistant ? 'model' : 'user', parts });
  }
  return out;
}

function toolsFor(wire: Wire, tools: readonly ClientTool[]): Json[] | null {
  if (tools.length === 0) return null;
  if (wire === 'openai') {
    return tools.map((t) => ({ type: 'function', function: { name: t.name, description: t.description, parameters: t.parameters } }));
  }
  if (wire === 'anthropic') {
    return tools.map((t) => ({ name: t.name, description: t.description, input_schema: t.parameters }));
  }
  return [{ functionDeclarations: tools.map((t) => ({ name: t.name, description: t.description, parameters: t.parameters })) }];
}

/** The client's tool_choice in the Anthropic spelling, from the OpenAI one. */
function toolChoiceToAnthropic(raw: Json): Json | string | null {
  const choice = raw.tool_choice;
  if (choice === 'auto') return { type: 'auto' };
  if (choice === 'required') return { type: 'any' };
  if (choice === 'none') return { type: 'none' };
  if (isRecord(choice) && isRecord(choice.function) && typeof choice.function.name === 'string') return { type: 'tool', name: choice.function.name };
  return null;
}

/** The client's tool_choice in the OpenAI spelling, from the Anthropic one. */
function toolChoiceToOpenAI(raw: Json): Json | string | null {
  const choice = raw.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') return 'auto';
    if (choice.type === 'any') return 'required';
    if (choice.type === 'none') return 'none';
    if (choice.type === 'tool' && typeof choice.name === 'string') return { type: 'function', function: { name: choice.name } };
  }
  return null;
}

function faithfulUpstream(view: ClientView, wire: Wire, model: string): Json {
  const raw = view.raw;
  if (wire === 'openai') {
    if (view.api === 'openai') {
      const body: Json = { ...raw, model };
      if (view.stream) body.stream_options = { ...(isRecord(raw.stream_options) ? raw.stream_options : {}), include_usage: true };
      return body;
    }
    const body: Json = { model, messages: messagesToOpenAI(view) };
    if (view.maxTokens !== null) body.max_tokens = view.maxTokens;
    if (view.temperature !== null) body.temperature = view.temperature;
    if (view.topP !== null) body.top_p = view.topP;
    if (view.stops.length > 0) body.stop = view.stops;
    const tools = toolsFor('openai', view.tools);
    if (tools !== null) body.tools = tools;
    const choice = toolChoiceToOpenAI(raw);
    if (choice !== null) body.tool_choice = choice;
    if (view.stream) {
      body.stream = true;
      body.stream_options = { include_usage: true };
    }
    return body;
  }
  if (wire === 'anthropic') {
    if (view.api === 'anthropic') return { ...raw, model };
    const body: Json = { model, messages: messagesToAnthropic(raw) };
    if (view.systemText !== '') body.system = view.systemText;
    if (view.maxTokens !== null) body.max_tokens = view.maxTokens;
    if (view.temperature !== null) body.temperature = view.temperature;
    if (view.topP !== null) body.top_p = view.topP;
    if (view.stops.length > 0) body.stop_sequences = view.stops;
    const tools = toolsFor('anthropic', view.tools);
    if (tools !== null) body.tools = tools;
    const choice = toolChoiceToAnthropic(raw);
    if (choice !== null) body.tool_choice = choice;
    if (view.stream) body.stream = true;
    return body;
  }
  // gemini
  const body: Json = { contents: messagesToGemini(view) };
  if (view.systemText !== '') body.systemInstruction = { parts: [{ text: view.systemText }] };
  const tools = toolsFor('gemini', view.tools);
  if (tools !== null) body.tools = tools;
  const config: Json = {};
  if (view.maxTokens !== null) config.maxOutputTokens = view.maxTokens;
  if (view.stops.length > 0) config.stopSequences = view.stops;
  if (view.temperature !== null) config.temperature = view.temperature;
  if (view.topP !== null) config.topP = view.topP;
  if (view.jsonMode) config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) body.generationConfig = config;
  return body;
}

// ---------------------------------------------------------------------------
// Canonical bodies: built from the ClientView alone, for when a faithful
// translation does not parse to the client's echo.
// ---------------------------------------------------------------------------

function canonicalUpstream(view: ClientView, wire: Wire, model: string): Json {
  if (wire === 'openai') {
    const messages: Json[] = [];
    if (view.systemText !== '') messages.push({ role: 'system', content: view.systemText });
    if (view.image === null) {
      messages.push({ role: 'user', content: view.userText });
    } else {
      const parts: Json[] = [];
      if (view.userText !== '') parts.push({ type: 'text', text: view.userText });
      parts.push(openAIImageBlock(view.image));
      messages.push({ role: 'user', content: parts });
    }
    const body: Json = { model, messages };
    const tools = toolsFor('openai', view.tools);
    if (tools !== null) body.tools = tools;
    if (view.maxTokens !== null) body.max_tokens = view.maxTokens;
    if (view.stops.length > 0) body.stop = view.stops;
    if (view.jsonMode) body.response_format = { type: 'json_object' };
    if (view.temperature !== null) body.temperature = view.temperature;
    if (view.topP !== null) body.top_p = view.topP;
    if (view.stream) {
      body.stream = true;
      body.stream_options = { include_usage: true };
    }
    return body;
  }
  if (wire === 'anthropic') {
    const blocks: Json[] = [];
    if (view.userText !== '' || view.image === null) blocks.push({ type: 'text', text: view.userText });
    if (view.image !== null) blocks.push(anthropicImageBlock(view.image));
    const body: Json = { model, messages: [{ role: 'user', content: blocks }] };
    if (view.systemText !== '') body.system = view.systemText;
    const tools = toolsFor('anthropic', view.tools);
    if (tools !== null) body.tools = tools;
    if (view.maxTokens !== null) body.max_tokens = view.maxTokens;
    if (view.stops.length > 0) body.stop_sequences = view.stops;
    if (view.temperature !== null) body.temperature = view.temperature;
    if (view.topP !== null) body.top_p = view.topP;
    if (view.stream) body.stream = true;
    return body;
  }
  const parts: Json[] = [];
  if (view.userText !== '' || view.image === null) parts.push({ text: view.userText });
  if (view.image !== null) parts.push(geminiImagePart(view.image));
  const body: Json = { contents: [{ role: 'user', parts }] };
  if (view.systemText !== '') body.systemInstruction = { parts: [{ text: view.systemText }] };
  const tools = toolsFor('gemini', view.tools);
  if (tools !== null) body.tools = tools;
  const config: Json = {};
  if (view.maxTokens !== null) config.maxOutputTokens = view.maxTokens;
  if (view.stops.length > 0) config.stopSequences = view.stops;
  if (view.temperature !== null) config.temperature = view.temperature;
  if (view.topP !== null) config.topP = view.topP;
  if (view.jsonMode) config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) body.generationConfig = config;
  return body;
}

/**
 * The upstream body for one provider wire: the faithful translation when the provider's parse of
 * it reproduces the client's echo exactly, the canonical body when it does not.
 */
export function buildUpstream(view: ClientView, wire: Wire, model: string): Json {
  const faithful = faithfulUpstream(view, wire, model);
  if (echoEquals(upstreamEcho(wire, faithful, model, view.stream), view.echo)) return faithful;
  return canonicalUpstream(view, wire, model);
}

// ---------------------------------------------------------------------------
// Provider answers, normalized.
// ---------------------------------------------------------------------------

interface SseFrame {
  event: string | null;
  data: string;
}

function sseFrames(text: string): SseFrame[] {
  const frames: SseFrame[] = [];
  for (const frame of text.replace(/\r\n/g, '\n').split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

function jsonOrNull(text: string): Json | null {
  try {
    const value: unknown = JSON.parse(text);
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function answerTextOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const parts: string[] = [];
  for (const block of content) if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') parts.push(block.text);
  return parts.join('');
}

/** Parse one provider answer (its wire, JSON or SSE) into the normalized shape. */
export function parseAnswer(wire: Wire, stream: boolean, bodyText: string): Normalized {
  if (wire === 'openai') {
    if (!stream) {
      const body = jsonOrNull(bodyText);
      if (body === null) return { text: '', input: null, output: null, stop: null };
      const choices = Array.isArray(body.choices) ? body.choices : [];
      const choice = isRecord(choices[0]) ? choices[0] : {};
      const message = isRecord(choice.message) ? choice.message : {};
      const usage = isRecord(body.usage) ? body.usage : {};
      return {
        text: answerTextOf(message.content),
        input: numberOrNull(usage.prompt_tokens),
        output: numberOrNull(usage.completion_tokens),
        stop: typeof choice.finish_reason === 'string' ? choice.finish_reason : null,
      };
    }
    let text = '';
    let input: number | null = null;
    let output: number | null = null;
    let stop: string | null = null;
    for (const frame of sseFrames(bodyText)) {
      const payload = frame.data.trim();
      if (payload === '[DONE]') continue;
      const chunk = jsonOrNull(payload);
      if (chunk === null) continue;
      if (isRecord(chunk.usage)) {
        input = numberOrNull(chunk.usage.prompt_tokens);
        output = numberOrNull(chunk.usage.completion_tokens);
      }
      const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
      const choice = isRecord(choices[0]) ? choices[0] : null;
      if (choice === null) continue;
      if (isRecord(choice.delta) && typeof choice.delta.content === 'string') text += choice.delta.content;
      if (typeof choice.finish_reason === 'string') stop = choice.finish_reason;
    }
    return { text, input, output, stop };
  }
  if (wire === 'anthropic') {
    if (!stream) {
      const body = jsonOrNull(bodyText);
      if (body === null) return { text: '', input: null, output: null, stop: null };
      const usage = isRecord(body.usage) ? body.usage : {};
      return {
        text: answerTextOf(body.content),
        input: numberOrNull(usage.input_tokens),
        output: numberOrNull(usage.output_tokens),
        stop: typeof body.stop_reason === 'string' ? body.stop_reason : null,
      };
    }
    let text = '';
    let input: number | null = null;
    let output: number | null = null;
    let stop: string | null = null;
    for (const frame of sseFrames(bodyText)) {
      const data = jsonOrNull(frame.data);
      if (data === null) continue;
      if (frame.event === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) {
        input = numberOrNull(data.message.usage.input_tokens);
      } else if (frame.event === 'content_block_delta' && isRecord(data.delta) && data.delta.type === 'text_delta' && typeof data.delta.text === 'string') {
        text += data.delta.text;
      } else if (frame.event === 'message_delta') {
        if (isRecord(data.usage)) output = numberOrNull(data.usage.output_tokens);
        if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') stop = data.delta.stop_reason;
      }
    }
    return { text, input, output, stop };
  }
  // gemini
  const fromBody = (body: Json | null): Normalized => {
    if (body === null) return { text: '', input: null, output: null, stop: null };
    const candidates = Array.isArray(body.candidates) ? body.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : {};
    const content = isRecord(candidate.content) ? candidate.content : {};
    const parts = Array.isArray(content.parts) ? content.parts : [];
    let text = '';
    for (const part of parts) if (isRecord(part) && typeof part.text === 'string') text += part.text;
    const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
    return {
      text,
      input: numberOrNull(usage.promptTokenCount),
      output: numberOrNull(usage.candidatesTokenCount),
      stop: typeof candidate.finishReason === 'string' ? candidate.finishReason : null,
    };
  };
  if (!stream) return fromBody(jsonOrNull(bodyText));
  const merged: Normalized = { text: '', input: null, output: null, stop: null };
  for (const frame of sseFrames(bodyText)) {
    const piece = fromBody(jsonOrNull(frame.data));
    merged.text += piece.text;
    if (piece.input !== null) merged.input = piece.input;
    if (piece.output !== null) merged.output = piece.output;
    if (piece.stop !== null) merged.stop = piece.stop;
  }
  return merged;
}

// ---------------------------------------------------------------------------
// Client answers, rendered from the normalized shape.
// ---------------------------------------------------------------------------

const STOP_FAMILIES = new Map<string, number>([
  ['stop', 0],
  ['end_turn', 0],
  ['STOP', 0],
  ['length', 1],
  ['max_tokens', 1],
  ['MAX_TOKENS', 1],
  ['tool_calls', 2],
  ['function_call', 2],
  ['tool_use', 2],
  ['content_filter', 3],
  ['refusal', 3],
  ['SAFETY', 3],
]);

const OPENAI_STOPS = ['stop', 'length', 'tool_calls', 'content_filter'];
const ANTHROPIC_STOPS = ['end_turn', 'max_tokens', 'tool_use', 'refusal'];

/** The client's token for a provider stop reason, by family. */
export function stopFor(api: 'openai' | 'anthropic', rawStop: string | null): string {
  const family = rawStop === null ? 0 : (STOP_FAMILIES.get(rawStop) ?? 0);
  return api === 'openai' ? OPENAI_STOPS[family] : ANTHROPIC_STOPS[family];
}

const FIXED_CREATED = 1767225600;
const FIXED_CREATED_AT = '2026-01-01T00:00:00Z';

export interface Rendered {
  status: number;
  contentType: string;
  body: string;
}

function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

function textPieces(text: string, size = 512): string[] {
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += size) pieces.push(text.slice(at, at + size));
  return pieces.length === 0 ? [''] : pieces;
}

/** Render the normalized answer in the client's wire, as JSON or as an SSE stream. */
export function renderAnswer(api: 'openai' | 'anthropic', model: string, answer: Normalized, stream: boolean): Rendered {
  const input = answer.input ?? 0;
  const output = answer.output ?? 0;
  if (api === 'openai') {
    const finish = stopFor('openai', answer.stop);
    if (!stream) {
      return {
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          id: 'chatcmpl-agg',
          object: 'chat.completion',
          created: FIXED_CREATED,
          model,
          choices: [{ index: 0, message: { role: 'assistant', content: answer.text }, finish_reason: finish, logprobs: null }],
          usage: { prompt_tokens: input, completion_tokens: output, total_tokens: input + output },
        }),
      };
    }
    const chunk = (delta: Json, finishReason: string | null): Json => ({
      id: 'chatcmpl-agg',
      object: 'chat.completion.chunk',
      created: FIXED_CREATED,
      model,
      choices: [{ index: 0, delta, finish_reason: finishReason }],
    });
    const frames: string[] = [sseEvent(null, chunk({ role: 'assistant', content: '' }, null))];
    for (const piece of textPieces(answer.text)) frames.push(sseEvent(null, chunk({ content: piece }, null)));
    frames.push(sseEvent(null, chunk({}, finish)));
    frames.push(
      sseEvent(null, {
        id: 'chatcmpl-agg',
        object: 'chat.completion.chunk',
        created: FIXED_CREATED,
        model,
        choices: [],
        usage: { prompt_tokens: input, completion_tokens: output, total_tokens: input + output },
      }),
    );
    frames.push('data: [DONE]\n\n');
    return { status: 200, contentType: 'text/event-stream', body: frames.join('') };
  }
  const stopReason = stopFor('anthropic', answer.stop);
  if (!stream) {
    return {
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        id: 'msg_agg',
        type: 'message',
        role: 'assistant',
        model,
        content: [{ type: 'text', text: answer.text }],
        stop_reason: stopReason,
        stop_sequence: null,
        usage: { input_tokens: input, output_tokens: output },
      }),
    };
  }
  const frames: string[] = [
    sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_agg',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: input, output_tokens: 0 },
      },
    }),
    sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }),
  ];
  for (const piece of textPieces(answer.text)) {
    frames.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: piece } }));
  }
  frames.push(
    sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }),
    sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stopReason, stop_sequence: null }, usage: { output_tokens: output } }),
    sseEvent('message_stop', { type: 'message_stop' }),
  );
  return { status: 200, contentType: 'text/event-stream', body: frames.join('') };
}

/** One model the entry serves, as its own GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
}

/** GET /v1/models in the OpenAI list shape. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return { object: 'list', data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_CREATED, owned_by: m.provider })) };
}

/** GET /v1/models in the Anthropic list shape. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.id, created_at: FIXED_CREATED_AT })),
    has_more: false,
  };
}

/** An error body in the client's wire. */
export function errorBody(api: 'openai' | 'anthropic', type: string, message: string): Json {
  if (api === 'anthropic') return { type: 'error', error: { type, message } };
  return { error: { message, type, param: null, code: null } };
}
