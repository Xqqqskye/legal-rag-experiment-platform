const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync('web/api-client.js', 'utf8');

function client(fetch, protocol = 'http:') {
  const window = { fetch, location: { protocol, href: 'http://127.0.0.1:8008/' } };
  const context = vm.createContext({ window, URL, AbortController, TypeError, setTimeout, clearTimeout });
  vm.runInContext(source, context);
  return window.apiFetch;
}

test('transient GET failures retry, message POSTs never auto-resubmit', async () => {
  let calls = 0;
  const get = client(async () => { if (++calls === 1) throw new TypeError('Failed to fetch'); return { ok: true }; });
  assert.equal((await get('/api/chat/sessions')).ok, true);
  assert.equal(calls, 2);
  calls = 0;
  const post = client(async () => { calls++; throw new TypeError('Failed to fetch'); });
  await assert.rejects(post('/api/chat/sessions/id/messages', { method: 'POST' }), /确认消息状态/);
  assert.equal(calls, 1);
});

test('validation errors are readable instead of object Object', async () => {
  const api = client(async () => ({ ok: false, status: 422, clone: () => ({ json: async () => ({ detail: [{ loc: ['body', 'message'], msg: '输入无效' }] }) }) }));
  await assert.rejects(api('/api/chat/sessions/id/messages', { method: 'POST' }), /message: 输入无效/);
});

test('file URLs explain the correct entry point without sending a request', async () => {
  const api = client(async () => { throw new Error('must not send'); }, 'file:');
  await assert.rejects(api('/api/chat/sessions'), /直接双击 HTML/);
});
