const input=document.getElementById('sourceText');
const button=document.getElementById('detectBtn');
const count=document.getElementById('charCount');
const clearButton=document.getElementById('clearBtn');
input.addEventListener('input',()=>count.textContent=`${input.value.length} / 2000`);
clearButton.addEventListener('click',()=>{input.value='';count.textContent='0 / 2000';input.focus();document.getElementById('analysis').classList.add('hidden');document.getElementById('emptyState').classList.remove('hidden')});
button.addEventListener('click',async()=>{
  if(!input.value.trim()){input.focus();input.setCustomValidity('Please enter some text first.');input.reportValidity();input.setCustomValidity('');return}
  button.disabled=true;button.querySelector('span:first-child').textContent='Analyzing…';
  try{
    const response=await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:input.value})});
    const data=await response.json();if(!response.ok)throw new Error(data.error||'Could not analyze this text.');
    document.getElementById('emptyState').classList.add('hidden');document.getElementById('analysis').classList.remove('hidden');
    document.getElementById('language').textContent=data.language;
    document.getElementById('confidence').textContent=`${(data.confidence*100).toFixed(2)}%`;
    document.getElementById('meterFill').style.width=`${Math.min(100,data.confidence*100)}%`;
    document.getElementById('preprocessed').textContent=data.preprocessed_text||'—';
    document.getElementById('modelUsed').textContent=data.model_used||'—';
    const notice=document.getElementById('notice');notice.textContent=data.message||'';notice.classList.toggle('hidden',!data.message);
  }catch(error){window.alert(error.message)}finally{button.disabled=false;button.querySelector('span:first-child').textContent='Identify language'}
});
