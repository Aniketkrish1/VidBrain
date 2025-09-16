const form = document.getElementById('startForm');
const jobBox = document.getElementById('job');
const statusEl = document.getElementById('status');
const bar = document.getElementById('bar');
const resultEl = document.getElementById('result');

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
    if(j.status === 'completed' && j.result){
      resultEl.classList.remove('hidden');
      resultEl.innerHTML = `<a class="btn" href="/api/download/${jobId}">Download summary</a>`;
      return;
    }
    if(j.status === 'failed'){
      resultEl.classList.remove('hidden');
      resultEl.innerHTML = `<span style="color:#ef4444">Failed: ${j.error||'unknown error'}</span>`;
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
  const r = await fetch('/api/start', {method:'POST', body: fd});
  if(!r.ok){
    const t = await r.json().catch(()=>({error:'Request failed'}));
    statusEl.textContent = t.error || 'Failed to start';
    return;
  }
  const {job_id} = await r.json();
  poll(job_id);
});


