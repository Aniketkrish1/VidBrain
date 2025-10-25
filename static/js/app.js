const form = document.getElementById('startForm');
const jobBox = document.getElementById('job');
const statusEl = document.getElementById('status');
const bar = document.getElementById('bar');
const resultEl = document.getElementById('result');
const progressPercent = document.getElementById('progress-percent');
const progressStage = document.getElementById('progress-stage');
const stageDetails = document.getElementById('stage-details');
const detailsEl = document.getElementById('details');

async function poll(jobId){
  try{
    const r = await fetch(`/api/status/${jobId}`);
    if(!r.ok){
      statusEl.textContent = 'Job not found';
      return;
    }
    const j = await r.json();
    const progress = j.progress || 0;
    bar.style.width = `${progress}%`;
    progressPercent.textContent = `${progress}%`;

    // Update progress stage and details
    if(j.stage) {
      progressStage.textContent = j.stage.charAt(0).toUpperCase() + j.stage.slice(1);
    }

    if(j.details) {
      detailsEl.textContent = j.details;
      stageDetails.classList.remove('hidden');
    } else {
      stageDetails.classList.add('hidden');
    }

    // Show detailed status information
    let statusText = `${j.status||'queued'}`;
    if(j.error) {
      statusText += `: ${j.error}`;
    }
    statusEl.textContent = statusText;

    if(j.status === 'completed' && j.result){
      let summariesHtml = '';
      if(j.summaries && j.summaries.length > 0) {
        summariesHtml = `
          <div style="margin-top: 20px; text-align: left;">
            <h4 style="color: var(--accent); margin-bottom: 10px;">📝 Generated Summaries:</h4>
            <div style="background: #0e141c; padding: 15px; border-radius: 8px; max-height: 300px; overflow-y: auto;">
              ${j.summaries.map(s => `
                <div style="margin-bottom: 15px; padding-bottom: 10px; border-bottom: 1px solid #374151;">
                  <div style="font-weight: bold; color: var(--accent); margin-bottom: 5px;">
                    Topic ${s.cluster_id + 1} (${s.start.toFixed(1)}s - ${s.end.toFixed(1)}s)
                  </div>
                  <div style="color: #e6eef8; line-height: 1.5;">
                    ${s.summary}
                  </div>
                </div>
              `).join('')}
            </div>
          </div>
        `;
      }

      resultEl.classList.remove('hidden');
      resultEl.innerHTML = `<div style="text-align: center; padding: 20px;">
                             <h3 style="color: var(--accent); margin-bottom: 15px;">✅ Video Generated Successfully!</h3>
                             <a class="btn" href="/api/download/${jobId}" style="font-size: 16px; padding: 15px 25px;">
                               📹 Download Summary Video (${(j.result && j.result.includes('summary_output.mp4')) ? 'Ready' : 'Processing...'})
                             </a>
                             <p style="margin-top: 15px; font-size: 0.9em; color: #666;">
                               📁 Saved to: <code style="background: #0e141c; padding: 2px 6px; border-radius: 3px;">${j.result}</code>
                             </p>
                             ${summariesHtml}
                           </div>`;
      return;
    }
    if(j.status === 'failed'){
      resultEl.classList.remove('hidden');
      resultEl.innerHTML = `<div style="color:#ef4444; padding: 10px; background: #fee; border-radius: 5px;">
                             ❌ Processing Failed: ${j.error||'unknown error'}
                           </div>`;
      return;
    }
    setTimeout(()=>poll(jobId), 1500); // More frequent updates for better UX
  }catch(e){
    setTimeout(()=>poll(jobId), 3000);
  }
}

form.addEventListener('submit', async (e)=>{
  e.preventDefault();
  const fd = new FormData(form);
  const simpleMode = document.getElementById('simple_mode').checked;
  
  // Validate inputs
  const youtubeUrl = fd.get('youtube_url');
  const videoFile = fd.get('video_file');
  const query = fd.get('query');
  
  if (!query || !query.trim()) {
    alert('Please enter a query (what topic you want to learn about)');
    return;
  }
  
  if (!youtubeUrl && (!videoFile || videoFile.size === 0)) {
    alert('Please provide either a YouTube URL or upload a video file');
    return;
  }
  
  jobBox.classList.remove('hidden');
  resultEl.classList.add('hidden');
  stageDetails.classList.add('hidden');

  // Initialize progress display
  bar.style.width = '0%';
  progressPercent.textContent = '0%';
  progressStage.textContent = 'Initializing';
  detailsEl.textContent = 'Starting processing...';
  statusEl.textContent = 'queued';

  if (simpleMode) {
    // Simple mode - direct summary
    try {
      progressStage.textContent = 'Generating Summary';
      detailsEl.textContent = 'Processing transcript and generating summary...';
      bar.style.width = '50%';
      progressPercent.textContent = '50%';
      
      const r = await fetch('/api/simple-summary', {method:'POST', body: fd});
      const result = await r.json();
      
      if (!r.ok || result.error) {
        statusEl.textContent = `Failed: ${result.error || 'Unknown error'}`;
        resultEl.classList.remove('hidden');
        resultEl.innerHTML = `<div style="color:#ef4444; padding: 10px; background: #fee; border-radius: 5px;">
                               ❌ Processing Failed: ${result.error || 'Unknown error'}
                             </div>`;
        return;
      }
      
      // Show success
      bar.style.width = '100%';
      progressPercent.textContent = '100%';
      progressStage.textContent = 'Completed';
      statusEl.textContent = 'completed';
      
      // Display simple summary result
      resultEl.classList.remove('hidden');
      resultEl.innerHTML = `
        <div style="color:#10b981; padding: 15px; background: #f0fff4; border-radius: 8px; margin-top: 20px;">
          <h4 style="color: var(--accent); margin-bottom: 10px;">📝 Generated Summary:</h4>
          <div style="background: #0e141c; padding: 15px; border-radius: 8px; color: #e6eef8; line-height: 1.6;">
            <div style="font-weight: bold; color: var(--accent); margin-bottom: 10px;">
              Topic: ${result.query}
            </div>
            <div>${result.summary}</div>
            <div style="margin-top: 10px; font-size: 0.9em; color: #9ca3af;">
              Processed ${result.transcript_length} sentences | Confidence: ${(result.confidence * 100).toFixed(0)}%
            </div>
          </div>
        </div>`;
        
    } catch (e) {
      statusEl.textContent = `Failed: ${e.message}`;
      resultEl.classList.remove('hidden');
      resultEl.innerHTML = `<div style="color:#ef4444; padding: 10px; background: #fee; border-radius: 5px;">
                             ❌ Network Error: ${e.message}
                           </div>`;
    }
  } else {
    // Full video mode - existing logic
    const r = await fetch('/api/start', {method:'POST', body: fd});
    if(!r.ok){
      const t = await r.json().catch(()=>({error:'Request failed'}));
      statusEl.textContent = t.error || 'Failed to start';
      return;
    }
    const {job_id} = await r.json();
    poll(job_id);
  }
});


