// Actual app.js in a minimal DOM: verifies workflow, not browser rendering/audio devices.
const fs=require('fs'),vm=require('vm'),assert=require('assert');
class Element {
  constructor(tag){this.tag=tag;this.children=[];this.events={};this.dataset={};this.className='';this.checked=false;this.value='';this.textContent='';this.classList={toggle:()=>{}};}
  append(...xs){this.children.push(...xs);} replaceChildren(...xs){this.children=xs;}
  addEventListener(name,fn){this.events[name]=fn;} setAttribute(k,v){this[k]=v;}
  click(){if(this.onclick)return this.onclick();} pause(){}
}
function setup(online=false){
  const ids={};for(const id of ['export','import','bundle','package','connection','progress','samples','content','sample-title','state','message','previous','next'])ids[id]=new Element('div');
  const tabs=['a','b'].map(k=>{const n=new Element('button');n.dataset.kind=k;return n;});
  function descend(n){return [n,...n.children.flatMap(descend)];}
  const elements=()=>[...Object.values(ids).flatMap(descend),...tabs];
  const storage=new Map();let exported,fail=false,server={schema:'pilot-annotations-v1',manifest_id:'test',revision:0,a:{},b:{}};
  const context={console,Date,JSON,Promise,Set,Map,Object,Array,Error,String,setTimeout,clearTimeout,
    location:{protocol:online?'http:':'file:'},confirm:()=>true,
    Blob:class{constructor(parts){exported=parts.join('');}},URL:{createObjectURL:()=> 'blob:test',revokeObjectURL:()=>{}},
    localStorage:{setItem:(k,v)=>storage.set(k,v),getItem:k=>storage.get(k)},
    fetch:async(url,options)=>{
      if(!options)return {ok:true,json:async()=>JSON.parse(JSON.stringify(server))};
      if(fail)return {ok:false,json:async()=>({error:'测试保存失败'})};
      const p=JSON.parse(options.body);assert.equal(p.expected_revision,server.revision);
      server={...p.annotations,revision:server.revision+1};return {ok:true,json:async()=>JSON.parse(JSON.stringify(server))};
    },
    document:{getElementById:id=>ids[id],createElement:tag=>new Element(tag),querySelectorAll:selector=>selector==='.tab'?tabs:elements().filter(x=>x.tag===selector)},
    window:{addEventListener:()=>{},PILOT_DATA:{manifest_id:'test',name:'ui-test',emotions:{happy:'开心',sad:'悲伤'},
      a:[{id:'A001',audio:'audio/a.wav',duration:5,transcript:'原转写'},{id:'A002',audio:'audio/a2.wav',duration:5,transcript:'第二条转写'}],
      b:[{id:'B001',text:'生成文本',target:'目标情绪',candidates:['A','B','C','D'].map(x=>({blind_id:x,audio:'audio/'+x+'.wav',duration:5}))}]}}
  };
  vm.createContext(context);vm.runInContext(fs.readFileSync('web/app.js','utf8'),context);
  function select(name,value){const n=elements().find(x=>x.tag==='input'&&x.name===name&&x.value===value);assert(n,name+':'+value);n.checked=true;n.events.change();}
  function text(value){const n=elements().find(x=>x.tag==='textarea'&&x['aria-label']==='核验转写');n.value=value;n.oninput();}
  return {ids,tabs,elements,storage,select,text,server:()=>server,fail:value=>{fail=value;},
    exported:()=>JSON.parse(exported),export:async()=>{await ids.export.onclick();return JSON.parse(exported);}};
}
async function main(){
  let h=setup();await new Promise(setImmediate);
  assert(!h.ids.save);assert.equal(h.elements().filter(x=>x.type==='checkbox').length,0);
  await h.ids.next.onclick();assert(h.ids.message.textContent.includes('请选择核验结论'));assert(h.ids['sample-title'].textContent.startsWith('A001'));
  h.select('emotion','sad');h.select('status','verified');h.text('更正后的转写');
  await h.ids.next.onclick();assert(h.ids['sample-title'].textContent.startsWith('A002'));
  let result=await h.export();assert(result.a.A001.final);assert.equal(result.a.A001.transcript,'更正后的转写');assert.equal(result.a.A001.completion_method,'navigation_save');
  await h.ids.previous.onclick();assert(h.ids['sample-title'].textContent.startsWith('A001'));
  h.text('返回后修改');await h.ids.next.onclick();result=await h.export();assert.equal(result.a.A001.transcript,'返回后修改');
  h.select('status','uncertain');await h.ids.next.onclick();assert(h.ids.message.textContent.includes('全部保存'));assert.equal(h.ids.next.textContent,'完成 A');
  await h.tabs[1].onclick();h.select('best','A');h.select('best','C');await h.ids.next.onclick();result=await h.export();
  assert.deepEqual(result.b.B001.best,['A','C']);assert(result.b.B001.final);
  h.select('status','all_tied');await h.ids.next.onclick();result=await h.export();assert.deepEqual(result.b.B001.best,['A','B','C','D']);
  h.select('status','all_bad');await h.ids.next.onclick();result=await h.export();assert.deepEqual(result.b.B001.best,[]);assert.equal(result.b.B001.status,'all_bad');
  assert(h.storage.has('bidirectional-pilot:test'));
  await h.ids.import.onchange({target:{files:[{text:async()=>JSON.stringify({...result,manifest_id:'wrong'})}],value:''}});assert(h.ids.message.textContent.includes('文件不属于'));
  result.b.B001={status:'uncertain',best:[],heard:true,final:true};
  await h.ids.import.onchange({target:{files:[{text:async()=>JSON.stringify(result)}],value:''}});result=await h.export();assert.equal(result.b.B001.status,'uncertain');
  h=setup(true);await new Promise(setImmediate);
  h.select('emotion','sad');h.select('status','verified');h.fail(true);
  await h.ids.next.onclick();assert(h.ids['sample-title'].textContent.startsWith('A001'));assert(h.ids.message.textContent.includes('保存失败'));
  h.fail(false);await h.ids.next.onclick();assert(h.ids['sample-title'].textContent.startsWith('A002'));assert(h.server().a.A001.final);
  await h.ids.previous.onclick();h.text('在线修改');await h.tabs[1].onclick();assert.equal(h.server().a.A001.transcript,'在线修改');
  await h.ids.next.onclick();assert(h.ids.message.textContent.includes('请选择最佳候选'));
  h.select('status','all_tied');await h.ids.next.onclick();assert.equal(h.server().b.B001.best.length,4);
  console.log('UI passed: next saves, previous editing, final item, partial drafts, tabs, no confirmation checkbox, all ties, import/export, online failure/retry.');
}
main().catch(e=>{console.error(e);process.exitCode=1;});
