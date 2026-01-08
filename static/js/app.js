const form = document.getElementById('startForm');
const jobBox = document.getElementById('job');
const statusEl = document.getElementById('status');
const bar = document.getElementById('bar');
const resultEl = document.getElementById('result');

// simple HTML escaper
function escapeHtml(unsafe){
  return String(unsafe)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

async function poll(jobId){
  try{
    const r = await fetch(`/api/status/${jobId}`);
    if(!r.ok){
      statusEl.textContent = 'Job not found';
      return;
    }
    const j = await r.json();
    bar.style.width = `${j.progress||0}%`;
    statusEl.textContent = `${j.status||'queued'}${j.error?': '+j.error:''}`;
    // Show friendly step hints
    const stepMap = {
      queued: 'Waiting to start',
      processing: 'Running pipeline (transcribing, clustering, summarizing)',
      completed: 'Completed',
      failed: 'Failed'
    };
  const hint = j.stage || stepMap[j.status] || '';
    const progressSteps = document.getElementById('progressSteps');
    progressSteps.textContent = hint;
    // toggle spinner
    const spinner = document.getElementById('spinner');
    if(j.status === 'processing'){
      spinner.classList.remove('hidden');
    } else {
      spinner.classList.add('hidden');
    }
    if(j.status === 'completed' && j.result){
      resultEl.classList.remove('hidden');
      // show inline player + download link
      resultEl.innerHTML = `
        <div>
          <video id="resultVideo" controls style="max-width:100%;height:auto">
            <source src="/api/download/${jobId}" type="video/mp4">
            Your browser does not support the video tag.
          </video>
        </div>
        <div style="margin-top:8px">
          <a class="btn download" href="/api/download/${jobId}">🎬 Download summary</a>
        </div>
      `;
      // Re-enable form
      document.getElementById('submitBtn').disabled = false;
      document.getElementById('clearBtn').disabled = false;
      // Fetch text summary and display if available (render as paragraph)
      try{
        const sres = await fetch(`/api/summary/${jobId}`);
        if(sres.ok){
          const summary = await sres.json();
          const summaryDiv = document.getElementById('textSummary');
          summaryDiv.classList.remove('hidden');
          let html = '<h4 style="margin-top:0">Video Summary</h4>';
          if(typeof summary === 'object'){
            const parts = [];
            if(Array.isArray(summary)){
              summary.forEach(s=>{ if(s && s.summary) parts.push(String(s.summary)); else if(s) parts.push(String(s)); });
            } else {
              const keys = Object.keys(summary).sort((a,b)=>Number(a)-Number(b));
              keys.forEach(k=>{ const item = summary[k]; if(item && item.summary) parts.push(String(item.summary)); });
            }
            const para = parts.join(' ');
            html += `<p>${escapeHtml(para)}</p>`;
          } else {
            html += `<p>${escapeHtml(String(summary))}</p>`;
          }
          html += '<div class="muted small" style="margin-top:8px">You can copy or download this text for notes.</div>';
          summaryDiv.innerHTML = html;
        }
      }catch(e){
        // ignore summary fetch errors
      }
      return;
    }
    if(j.status === 'failed'){
      resultEl.classList.remove('hidden');
      resultEl.innerHTML = `<span style="color:#ef4444">Failed: ${j.error||'unknown error'}</span>`;
      document.getElementById('submitBtn').disabled = false;
      document.getElementById('clearBtn').disabled = false;
      return;
    }
    setTimeout(()=>poll(jobId), 2500);
  }catch(e){
    setTimeout(()=>poll(jobId), 3000);
  }
}

form.addEventListener('submit', async (e)=>{
  e.preventDefault();
  const fd = new FormData(form);
  jobBox.classList.remove('hidden');
  resultEl.classList.add('hidden');
  bar.style.width = '0%';
  statusEl.textContent = 'queued';
  // disable UI while job runs
  document.getElementById('submitBtn').disabled = true;
  document.getElementById('clearBtn').disabled = true;
  const r = await fetch('/api/start', {method:'POST', body: fd});
  if(!r.ok){
    const t = await r.json().catch(()=>({error:'Request failed'}));
    statusEl.textContent = t.error || 'Failed to start';
    document.getElementById('submitBtn').disabled = false;
    document.getElementById('clearBtn').disabled = false;
    return;
  }
  const {job_id} = await r.json();
  poll(job_id);
});

// Clear form button
document.getElementById('clearBtn').addEventListener('click', ()=>{
  form.reset();
});


