// Same-origin proxy to the persistent GPU worker; no credentials in the browser.
const panel = document.createElement('section');
panel.innerHTML = '<h2>Streaming Render</h2><small id="render-status">GPU worker status</small><img id="render-feedback" alt="Streaming H3 video feedback" style="display:none;width:100%;border-radius:8px;margin-top:10px">';
document.querySelector('main').append(panel);
const status = panel.querySelector('#render-status');
const feedback = panel.querySelector('#render-feedback');
let timer;

async function request(path, options) {
  const response = await fetch('/runtime/' + path, options);
  const value = await response.json();
  if (!response.ok) throw new Error(value.error || response.statusText);
  return value;
}

const WINDOW = 10;
const history = [];
const total = {chunks: 0, frames: 0, timedFrames: 0, intervalMs: 0};
let lastRound = -1;
status.style.whiteSpace = 'pre-line';

function resetStats() {
  history.length = 0;
  Object.assign(total, {chunks: 0, frames: 0, timedFrames: 0, intervalMs: 0});
  lastRound = -1;
}

function ms(value) {
  return value == null ? '-' : value.toFixed(0) + ' ms';
}

function fpsText(value) {
  return value == null ? '- fps' : value.toFixed(1) + ' fps';
}

function median(values) {
  const sorted = values.filter(value => value != null).sort((a, b) => a - b);
  if (!sorted.length) return null;
  const mid = sorted.length >> 1;
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

// Sustained fps is output frames per publication interval; the first chunk has no interval yet,
// so its frames are excluded from the timed total.
function record(metrics) {
  if (metrics.round === lastRound) return;
  lastRound = metrics.round;
  const interval = metrics.publication_interval_ms;
  history.push({...metrics, fps: interval ? metrics.output_frames * 1000 / interval : null});
  if (history.length > WINDOW) history.shift();
  total.chunks += 1;
  total.frames += metrics.output_frames;
  if (interval) {
    total.timedFrames += metrics.output_frames;
    total.intervalMs += interval;
  }
}

function line(label, pick) {
  return label + ' · ' + fpsText(pick('fps')) + ' · RGB-ready ' + ms(pick('rgb_ready_wall_ms'))
    + ' (encode ' + ms(pick('encode_ms')) + ', render ' + ms(pick('render_ms'))
    + ', decode ' + ms(pick('decode_ms')) + ')'
    + ' · interval ' + ms(pick('publication_interval_ms'))
    + ' · input to video ' + ms(pick('oldest_input_to_rgb_ready_ms'));
}

function formatMetrics() {
  const now = history[history.length - 1];
  const average = total.intervalMs ? total.timedFrames * 1000 / total.intervalMs : null;
  return line('Chunk ' + now.round + ' (' + now.output_frames + ' frames)', key => now[key]) + '\n'
    + line('Median of last ' + history.length, key => median(history.map(entry => entry[key]))) + '\n'
    + 'Total · ' + total.chunks + ' chunks · ' + total.frames + ' frames · average ' + fpsText(average)
    + ' over ' + (total.intervalMs / 1000).toFixed(1) + ' s';
}

export async function startRuntime(config) {
  const health = await request('health');
  if (!health.ready) throw new Error(health.error || 'GPU worker loading / warming up');
  const session = await request('session', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(config),
  });
  const deadline = performance.now() + 180000;
  for (;;) {
    const value = await request('status/' + session.id);
    if (value.error || value.closed) throw new Error(value.error || 'Session closed');
    if (value.ready) break;
    if (performance.now() > deadline) {
      await closeRuntime(session.id);
      throw new Error('Session setup timed out');
    }
    await new Promise(resolve => setTimeout(resolve, 250));
  }
  feedback.src = '/runtime/stream/' + session.id;
  feedback.style.display = 'block';
  resetStats();
  clearInterval(timer);
  timer = setInterval(async () => {
    try {
      const value = await request('status/' + session.id);
      const metrics = value.metrics;
      if (!value.error && metrics.backend !== 'passthrough' && metrics.rgb_ready_wall_ms != null) record(metrics);
      status.textContent = value.error || (metrics.backend === 'passthrough'
        ? 'Transport diagnostic · chunk ' + metrics.round
        : metrics.rgb_ready_wall_ms == null
        ? 'Waiting for first reference chunk'
        : formatMetrics());
      if (value.closed) clearInterval(timer);
    } catch (error) {
      status.textContent = error.message;
    }
  }, 500);
  return {...session, runtime: true};
}

export async function sendRuntimeFrame(session, index, blob, controls) {
  // Retry the identical frame; never skip simulation steps on backpressure.
  for (;;) {
    const response = await fetch('/runtime/frame/' + session.id + '/' + index, {
      method: 'POST', body: blob,
      headers: {'Content-Type': 'image/png', 'X-Controls': JSON.stringify(controls)},
    });
    if (response.status === 429) {
      const state = await request('status/' + session.id);
      if (state.closed || state.error) throw new Error(state.error || 'Session closed');
      await new Promise(resolve => setTimeout(resolve, 25));
      continue;
    }
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || 'Reference frame rejected');
    return result;
  }
}

export async function closeRuntime(id) {
  clearInterval(timer);
  if (id) await request('close/' + id, {method: 'POST', body: ''});
  feedback.removeAttribute('src');
}
