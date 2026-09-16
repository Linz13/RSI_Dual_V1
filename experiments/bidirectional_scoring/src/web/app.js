"use strict";
(() => {
  const data=window.PILOT_DATA, $=id=>document.getElementById(id);
  const cacheKey="bidirectional-pilot:"+data.manifest_id;
  let kind=data.a.length?"a":"b", index=0, online=location.protocol!=="file:", busy=false;
  let state={schema:"pilot-annotations-v1",manifest_id:data.manifest_id,revision:0,a:{},b:{}};
  let chain=Promise.resolve(), timer, syncing=false, unsaved=false;
  const el=(tag,text,cls)=>{const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n;};
  const current=()=>data[kind][index];
  const entry=()=>current()?(state[kind][current().id]||{}):{};
  function message(text,error=false){$("message").textContent=text;$("message").className=error?"error-text":"";}
  function connection(text,error=false){$("connection").textContent=text;$("connection").className="notice"+(error?" error":"");}
  function localSave(){
    try{localStorage.setItem(cacheKey,JSON.stringify(state));}
    catch(e){connection("浏览器暂存不可用，请及时导出标注。"+(online?" 服务器保存仍可使用。":""),true);}
  }
  function changed(){state.updated_at=new Date().toISOString();unsaved=true;localSave();}
  async function push(){
    if(!online)return;
    syncing=true;
    const payload=JSON.parse(JSON.stringify(state));
    try{
      const response=await fetch("/api/annotations",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({annotations:payload,expected_revision:state.revision})});
      const result=await response.json();
      if(!response.ok)throw Error(result.error||"保存失败");
      state.revision=result.revision;
      if(state.updated_at===payload.updated_at)unsaved=false;
      localSave();connection("已保存到服务器 · "+new Date().toLocaleTimeString());
    }catch(e){connection("服务器未保存："+e.message+"。草稿已保留，请重试或导出备份。",true);throw e;}
    finally{syncing=false;}
  }
  function schedule(immediate=false){
    localSave();clearTimeout(timer);
    if(!online){connection("离线模式 · 标注暂存本浏览器，结束时请导出，再导入服务器页面。");return Promise.resolve();}
    const queue=()=>{chain=chain.catch(()=>{}).then(push);return chain;};
    if(immediate)return queue();
    timer=setTimeout(()=>queue().catch(()=>{}),650);return Promise.resolve();
  }
  function change(patch){
    if(busy)return;
    state[kind][current().id]={...entry(),...patch,final:false,updated_at:new Date().toISOString()};
    changed();schedule();navigation();$("state").textContent="草稿 · 下一条时保存为已完成";message("");
  }
  function valid(k,v,row){
    if(!v||typeof v!=="object")return "标注内容格式错误。";
    if(k==="a"){
      if(!["verified","uncertain","unusable"].includes(v.status))return "请选择核验结论；无法判断时可以直接选相应选项。";
      if(v.status==="verified"&&(!data.emotions[v.emotion]||!v.transcript?.trim()))return "请填写明确情绪和核验转写；听不出来时请选择无法判定。";
    }else{
      if(!["preferred","all_bad","uncertain"].includes(v.status))return "请选择最佳候选、全部不合格或无法判断。";
      const best=v.best||[],ids=new Set(row.candidates.map(c=>c.blind_id));
      if(!Array.isArray(best)||best.some(x=>!ids.has(x))||best.length!==new Set(best).size)return "候选选择无效。";
      if((v.status==="preferred")!==!!best.length)return "请选择最佳候选；全部不合格／无法判断时不要勾选候选。";
    }return "";
  }
  async function finishCurrent(required){
    if(!current())return true;
    const v={...entry()};if(kind==="a"&&v.transcript===undefined)v.transcript=current().transcript;
    const error=valid(kind,v,current());
    if(error&&required){message(error,true);return false;}
    if(!error){
      state[kind][current().id]={...v,final:true,heard:true,completion_method:"navigation_save",updated_at:new Date().toISOString()};
      changed();
    }
    await schedule(true);return true;
  }
  async function navigate(nextKind,nextIndex,required=false){
    if(busy)return;
    busy=true;$("previous").disabled=true;$("next").disabled=true;
    try{
      if(!await finishCurrent(required))return;
      const last=kind===nextKind&&index===nextIndex;
      kind=nextKind;index=nextIndex;render();
      if(last&&required){
        const done=data[kind].filter(r=>state[kind][r.id]?.final).length;
        message(done===data[kind].length?kind.toUpperCase()+" 已全部保存。可切换另一方向或导出标注。":"本条已保存，本方向还有 "+(data[kind].length-done)+" 条待完成，可从左侧返回。");
      }
    }catch(e){message("保存失败，已停留在当前样本。请重试；也可先导出备份。",true);}
    finally{busy=false;$("previous").disabled=!data[kind].length||index===0;$("next").disabled=!data[kind].length;}
  }
  function option(parent,name,value,title,checked,onChange,type="radio"){
    const label=el("label",undefined,"option"),input=el("input");input.type=type;input.name=name;input.value=value;input.checked=checked;
    input.addEventListener("change",()=>{if(!busy)onChange(input.checked);});label.append(input,el("span",title));parent.append(label);return input;
  }
  function field(title){const f=el("fieldset");f.append(el("legend",title));$("content").append(f);return f;}
  function audio(parent,path,title){
    const box=el("div",undefined,"audio-card");box.append(el("strong",title));const player=el("audio");player.controls=true;player.preload="metadata";player.src=path;
    player.addEventListener("play",()=>document.querySelectorAll("audio").forEach(a=>{if(a!==player)a.pause();}));
    player.addEventListener("error",()=>message("音频打开失败。离线使用请完整解压，并保留 audio 文件夹。",true));box.append(player);parent.append(box);return box;
  }
  function navigation(){
    document.querySelectorAll(".tab").forEach(x=>{x.classList.toggle("active",x.dataset.kind===kind);x.setAttribute("aria-current",x.dataset.kind===kind?"page":"false");});
    const rows=data[kind],done=rows.filter(r=>state[kind][r.id]?.final).length;
    $("progress").textContent=kind.toUpperCase()+" 已完成 "+done+" / "+rows.length;
    $("samples").replaceChildren();rows.forEach((r,i)=>{const b=el("button",r.id,"sample"+(state[kind][r.id]?.final?" done":"")+(i===index?" selected":""));b.onclick=()=>navigate(kind,i);$("samples").append(b);});
  }
  function render(){
    navigation();const rows=data[kind];$("content").replaceChildren();message("");
    ["next","previous"].forEach(id=>$(id).disabled=!rows.length);
    if(!rows.length){$("sample-title").textContent="此方向暂无样本";$("state").textContent="";return;}
    const r=current(),v=entry();$("sample-title").textContent=r.id+" · "+(kind==="a"?"核验真实语音":"比较生成语音");$("state").textContent=v.final?"已完成 · 可修改":"待标注";
    if(kind==="a"){
      $("content").append(el("div","依据声音判断情绪，核对并修改转写。明确听到平静可以选平静；听不出来、混合情绪请选择无法判定。填好后点击下一条即可保存。","instruction"));
      audio($("content"),r.audio,"真实语音 · "+r.duration.toFixed(1)+" 秒");
      let f=field("1. 声音表达的情绪");Object.entries(data.emotions).forEach(([key,label])=>option(f,"emotion",key,label,v.emotion===key,()=>change({emotion:key})));
      f=field("2. 核对转写（可直接修改）");const text=el("textarea");text.value=v.transcript??r.transcript;text.setAttribute("aria-label","核验转写");text.oninput=()=>change({transcript:text.value});f.append(text);
      f=field("3. 核验结论");[["verified","情绪明确，转写已核对"],["uncertain","混合情绪／无法判定"],["unusable","音频损坏／无法可靠转写"]].forEach(([key,label])=>option(f,"status",key,label,v.status===key,()=>change({status:key})));
    }else{
      const help=el("div",undefined,"instruction");help.append(el("p",r.target,"target"),el("p","生成文本："+r.text),el("p","听完全部四条，选最符合目标情绪的候选；多选表示并列最佳。都合格但有优劣时只选更好的；四条一样好可选“四条并列最佳”。"));$("content").append(help);
      const grid=el("div",undefined,"candidates");$("content").append(grid);
      r.candidates.forEach(c=>{const box=audio(grid,c.audio,"候选 "+c.blind_id+" · "+c.duration.toFixed(1)+" 秒");option(box,"best",c.blind_id,"最符合目标（可并列）",(v.best||[]).includes(c.blind_id),checked=>{
        const best=(entry().best||[]).filter(x=>x!==c.blind_id);if(checked)best.push(c.blind_id);
        change({best,status:best.length?"preferred":""});render();
      },"checkbox");});
      const f=field("整体判断（选了候选即表示有最佳；以下可直接选择）");
      [["all_tied","四条并列最佳，都符合且无明显优劣"],["all_bad","全部不合格"],["uncertain","无法判断"]].forEach(([key,label])=>{
        const checked=key==="all_tied"?v.status==="preferred"&&(v.best||[]).length===r.candidates.length:v.status===key;
        option(f,"status",key,label,checked,()=>{change(key==="all_tied"?{status:"preferred",best:r.candidates.map(c=>c.blind_id)}:{status:key,best:[]});render();});
      });
    }
    const notes=field("备注（可选）"),ta=el("textarea",undefined,"notes");ta.value=v.notes||"";ta.setAttribute("aria-label","备注");ta.oninput=()=>change({notes:ta.value});notes.append(ta);
    $("content").append(el("p",kind==="a"?"点击下一条表示已完成本条听音和转写核验。":"请听完全部候选再点下一条。选不出时可以保留并列或无法判断。","help"));
    $("previous").disabled=index===0;$("next").textContent=index===rows.length-1?"完成 "+kind.toUpperCase():"下一条";
  }
  $("previous").onclick=()=>navigate(kind,Math.max(0,index-1));
  $("next").onclick=()=>navigate(kind,Math.min(data[kind].length-1,index+1),true);
  document.querySelectorAll(".tab").forEach(b=>b.onclick=()=>navigate(b.dataset.kind,0));
  $("export").onclick=async()=>{
    if(busy)return;
    try{await finishCurrent(false);}catch(e){message("服务器未保存，已导出浏览器中的备份。",true);}
    localSave();const blob=new Blob([JSON.stringify(state,null,2)],{type:"application/json"}),a=el("a");a.href=URL.createObjectURL(blob);a.download=data.name+"_annotations.json";a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);
  };
  $("import").onchange=async e=>{
    if(busy)return;
    const file=e.target.files[0];if(!file)return;
    try{
      const incoming=JSON.parse(await file.text());if(incoming.schema!==state.schema||incoming.manifest_id!==data.manifest_id)throw Error("文件不属于此样本包");
      for(const k of ["a","b"]){
        const rows=new Map(data[k].map(r=>[r.id,r]));
        if(!incoming[k]||typeof incoming[k]!=="object"||Array.isArray(incoming[k])||Object.keys(incoming[k]).some(id=>!rows.has(id)))throw Error("文件含未知样本");
        for(const [rid,v] of Object.entries(incoming[k]))if(v.final&&(valid(k,v,rows.get(rid))||!v.heard))throw Error("已完成标注缺少有效判断："+rid);
      }
      if(!confirm("导入将替换文件中同编号样本的标注，其余样本保留。是否继续？"))return;
      state.a={...state.a,...incoming.a};state.b={...state.b,...incoming.b};changed();await schedule(true);render();message("标注已导入。");
    }catch(error){message("导入失败："+error.message,true);}finally{e.target.value="";}
  };
  window.addEventListener("beforeunload",e=>{if(online&&(syncing||unsaved)){e.preventDefault();e.returnValue="";}localSave();});
  async function start(){
    $("package").textContent=data.name+" · A "+data.a.length+" 条 / B "+data.b.length+" 组";
    $("bundle").hidden=!online;let saved=null;try{saved=JSON.parse(localStorage.getItem(cacheKey)||"null");}catch(e){}
    if(online){
      try{
        const response=await fetch("/api/annotations");if(!response.ok)throw Error("HTTP "+response.status);state=await response.json();
        if(state.manifest_id!==data.manifest_id)throw Error("服务器样本包与页面不同，请刷新");
        connection("服务器模式 · 填好后点下一条，自动保存为已完成。可返回修改。");
        if(saved&&saved.updated_at>state.updated_at&&confirm("本浏览器有较新的草稿。是否恢复草稿？")){state.a={...state.a,...saved.a};state.b={...state.b,...saved.b};changed();await schedule(true);}
      }catch(e){online=false;if(saved)state=saved;connection("无法连接标注服务，已转为浏览器暂存。请导出备份，稍后导入服务器。",true);}
    }else{if(saved)state=saved;connection("离线模式 · 完整解压后可播放；点下一条保存，结束时请导出标注。");}
    render();
  }
  start();
})();
