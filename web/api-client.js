/* Same-origin API transport with actionable errors and bounded read retries. */
(function (root) {
  function detailMessage(detail) {
    if (Array.isArray(detail)) {
      return detail.map(item => `${(item.loc || []).filter(key => key !== 'body').join('.')}: ${item.msg || '输入无效'}`).join('；');
    }
    return typeof detail === 'string' ? detail : '';
  }

  async function apiFetch(path, options = {}) {
    if (!/^https?:$/.test(root.location.protocol)) {
      throw new Error('请通过 http://127.0.0.1:8008/ 打开 Demo；直接双击 HTML 文件无法连接会话接口。');
    }
    const url = new URL(path, root.location.href);
    const method = (options.method || 'GET').toUpperCase();
    const retries = method === 'GET' ? 1 : 0;
    for (let attempt = 0; attempt <= retries; attempt += 1) {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), method === 'GET' ? 15000 : 240000);
      try {
        const response = await root.fetch(url.href, { ...options, cache: 'no-store', signal: controller.signal });
        if (!response.ok) {
          let detail = '';
          try { const data = await response.clone().json(); detail = detailMessage(data.detail || data.error); } catch (_) { /* non-JSON response */ }
          throw new Error(detail || `接口 ${url.pathname} 返回 HTTP ${response.status}`);
        }
        return response;
      } catch (error) {
        const network = error instanceof TypeError || error.name === 'AbortError';
        if (network && attempt < retries) {
          await new Promise(resolve => setTimeout(resolve, 500));
          continue;
        }
        if (network) {
          const reason = error.name === 'AbortError' ? '请求超时' : '浏览器未能连接服务';
          const next = method === 'GET' ? '请打开上面的接口地址检查连接。' : '请先重新加载当前会话确认消息状态，再决定是否重发。';
          throw new Error(`${reason}：${url.href}。${next}`, { cause: error });
        }
        throw error;
      } finally {
        clearTimeout(timer);
      }
    }
  }
  root.apiFetch = apiFetch;
})(window);
