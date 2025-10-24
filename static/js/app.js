const form = document.getElementById('startForm');
const uploadForm = document.getElementById('uploadForm');
const jobBox = document.getElementById('job');
const statusEl = document.getElementById('status');
const bar = document.getElementById('bar');
const resultEl = document.getElementById('result');
const progressPercent = document.getElementById('progress-percent');
const progressStage = document.getElementById('progress-stage');
const stageDetails = document.getElementById('stage-details');
const detailsEl = document.getElementById('details');
const summarySection = document.getElementById('summary-section');
const summaryText = document.getElementById('summary-text');
const summaryQuery = document.getElementById('summary-query');
const summaryContainer = document.getElementById('summary-container');

// Tab switching
function switchTab(tab) {
  const youtubeTab = document.getElementById('youtubeTab');
  const uploadTab = document.getElementById('uploadTab');
  const youtubeForm = document.getElementById('startForm');
  const uploadFormEl = document.getElementById('uploadForm');
  
  if (tab === 'youtube') {
    youtubeTab.classList.add('active');
    youtubeTab.style.color = 'var(--accent)';
    youtubeTab.style.borderBottom = '2px solid var(--accent)';
    uploadTab.classList.remove('active');
    uploadTab.style.color = '#888';
    uploadTab.style.borderBottom = '2px solid transparent';
    youtubeForm.classList.remove('hidden');
    uploadFormEl.classList.add('hidden');
  } else {
    uploadTab.classList.add('active');
    uploadTab.style.color = 'var(--accent)';
    uploadTab.style.borderBottom = '2px solid var(--accent)';
    youtubeTab.classList.remove('active');
    youtubeTab.style.color = '#888';
    youtubeTab.style.borderBottom = '2px solid transparent';
    uploadFormEl.classList.remove('hidden');
    youtubeForm.classList.add('hidden');
  }
}

// Make switchTab globally available
window.switchTab = switchTab;

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

    // Display summary when available
    if(j.summaries && j.summaries.length > 0 && j.summaries[0].summary) {
      const summary = j.summaries[0];
      summarySection.classList.remove('hidden');
      
      if(summary.query) {
        summaryQuery.innerHTML = `<strong>Topic:</strong> ${summary.query}`;
      } else {
        summaryQuery.innerHTML = '';
      }
      
      summaryText.innerHTML = summary.summary.replace(/\n/g, '<br>');
      
      // Add timestamp info if available
      if(summary.start !== undefined && summary.end !== undefined) {
        summaryQuery.innerHTML += ` <span style="color: #666;">(${summary.start.toFixed(1)}s - ${summary.end.toFixed(1)}s)</span>`;
      }
    }

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
                    ${s.query ? `Topic: ${s.query}` : `Topic ${s.cluster_id + 1}`} (${s.start.toFixed(1)}s - ${s.end.toFixed(1)}s)
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
                             <video controls style="max-width: 100%; border-radius: 8px; margin: 20px 0;">
                               <source src="/api/download/${jobId}" type="video/mp4">
                               Your browser does not support the video tag.
                             </video>
                             <a class="btn" href="/api/download/${jobId}" style="font-size: 16px; padding: 15px 25px;">
                               � Download Summary Video
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
  jobBox.classList.remove('hidden');
  resultEl.classList.add('hidden');
  stageDetails.classList.add('hidden');
  summarySection.classList.add('hidden'); // Hide summary initially

  // Initialize progress display
  bar.style.width = '0%';
  progressPercent.textContent = '0%';
  progressStage.textContent = 'Initializing';
  detailsEl.textContent = 'Starting video processing...';
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

// Upload form handler
uploadForm.addEventListener('submit', async (e)=>{
  e.preventDefault();
  const fd = new FormData(uploadForm);
  
  // Check file size
  const fileInput = uploadForm.querySelector('input[type="file"]');
  const file = fileInput.files[0];
  
  if (!file) {
    statusEl.textContent = 'Please select a video file';
    return;
  }
  
  // Warn if file is too large
  const fileSizeMB = file.size / 1024 / 1024;
  if (fileSizeMB > 500) {
    if (!confirm(`File size is ${fileSizeMB.toFixed(1)}MB. Large files may take a long time to process. Continue?`)) {
      return;
    }
  }
  
  jobBox.classList.remove('hidden');
  resultEl.classList.add('hidden');
  stageDetails.classList.add('hidden');
  summarySection.classList.add('hidden');

  // Initialize progress display
  bar.style.width = '0%';
  progressPercent.textContent = '0%';
  progressStage.textContent = 'Uploading video...';
  detailsEl.textContent = `Uploading ${file.name} (${fileSizeMB.toFixed(1)}MB)...`;
  stageDetails.classList.remove('hidden');
  statusEl.textContent = 'uploading';

  const r = await fetch('/api/upload', {method:'POST', body: fd});
  if(!r.ok){
    const t = await r.json().catch(()=>({error:'Upload failed'}));
    statusEl.textContent = t.error || 'Upload failed';
    return;
  }
  const {job_id} = await r.json();
  poll(job_id);
});



