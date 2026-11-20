/* CIQ-IR exact-decimal core. No physical G, c, h, or ħ is hard-coded.
 * The reference 10 µm and SI unit conversions are not fitted constants.
 * BigInt rationals preserve 10^-19 relative differences.
 */
(function(root) {
'use strict';
const ten = e => 10n ** BigInt(e);
function gcd(a, b) { a = a < 0n ? -a : a; b = b < 0n ? -b : b; while (b) [a,b] = [b,a%b]; return a; }
class Q {
  constructor(n,d=1n) {
    n=BigInt(n); d=BigInt(d);
    if(d===0n) throw Error('Division durch null');
    if(d<0n){n=-n;d=-d;}
    const g=gcd(n,d);this.n=n/g;this.d=d/g;
  }
  static from(v) {
    if(v instanceof Q)return v;
    const txt=String(v).trim();
    const m=txt.match(/^([+-]?)(\d+(?:\.\d*)?|\.\d+)(?:[eE]([+-]?\d+))?$/);
    if(!m)throw Error('Ungültige Dezimalzahl: '+txt);
    const parts=m[2].split('.'), digits=parts.join(''), exp=Number(m[3]||0)-(parts[1]||'').length;
    if(!Number.isSafeInteger(exp)||Math.abs(exp)>500)throw Error('Exponent außerhalb Eingabebereich');
    const val=BigInt(digits)*(m[1]==='-'?-1n:1n);
    return exp>=0?new Q(val*ten(exp)):new Q(val,ten(-exp));
  }
  add(v){v=Q.from(v);return new Q(this.n*v.d+v.n*this.d,this.d*v.d);}
  sub(v){v=Q.from(v);return new Q(this.n*v.d-v.n*this.d,this.d*v.d);}
  mul(v){v=Q.from(v);return new Q(this.n*v.n,this.d*v.d);}
  div(v){v=Q.from(v);return new Q(this.n*v.d,this.d*v.n);}
  abs(){return new Q(this.n<0n?-this.n:this.n,this.d);}
  isPositive(){return this.n>0n;}
  toFixed(digits=32){
    if(digits<0||digits>120)throw Error('Unzulässige Stellenzahl');
    const sign=this.n<0n?'-':'', a=this.n<0n?-this.n:this.n;
    const whole=a/this.d, frac=((a%this.d)*ten(digits)/this.d).toString().padStart(digits,'0');
    return sign+whole.toString()+(digits?'.'+frac.replace(/0+$/,''):'');
  }
  toScientific(digits=18){
    if(this.n===0n)return '0';
    let n=this.n<0n?-this.n:this.n,d=this.d;
    let exp=n.toString().length-d.toString().length;
    if((exp>=0&&n<d*ten(exp))||(exp<0&&n*ten(-exp)<d))exp--;
    let shift=digits-1-exp;
    let m=shift>=0?n*ten(shift)/d:n/(d*ten(-shift));
    const s=m.toString().padStart(digits,'0');
    return (this.n<0n?'-':'')+s[0]+'.'+s.slice(1)+'e'+(exp>=0?'+':'')+exp;
  }
}
const q=Q.from,ref=q('10').mul(q('1e-6'));
const read=(input,key)=>{
 const value=(input[key]??'').toString().trim();return value===''?null:q(value);
};
function positive(x,key){if(x!==null&&!x.isPositive())throw Error(key+' muss > 0 sein');}
function invert(delta,rA,rB){
 const k=q(1).add(delta), k2=k.mul(k), den=k2.div(rA).sub(q(1).div(rB));
 if(!k.isPositive())throw Error('K_IR muss > 0 sein');
 return k2.sub(1).div(den);
}
function slope(delta,rA,rB){
 const k=q(1).add(delta), den=k.mul(k).div(rA).sub(q(1).div(rB));
 return q(2).mul(k).mul(q(1).div(rA).sub(q(1).div(rB))).div(den.mul(den));
}
function evaluate(input){
 const delta=read(input,'delta_k'),sigma=read(input,'sigma_delta_k'),rA=read(input,'r_a_m'),rB=read(input,'r_b_m');
 const mass=read(input,'mass_kg'),lambdaC=read(input,'calibration_lambda_m'),frequencyC=read(input,'calibration_frequency_hz'),gRef=read(input,'comparison_g_only');
 [ [rA,'r_a_m'],[rB,'r_b_m'],[mass,'mass_kg'],[lambdaC,'calibration_lambda_m'],[frequencyC,'calibration_frequency_hz'] ].forEach(([x,k])=>positive(x,k));
 if(sigma!==null&&sigma.n<0n)throw Error('sigma_delta_k muss >= 0 sein');
 const source=input.mass_provenance||'unknown',vac=input.vacuum_provenance||'unknown',statuses=[];
 if(!['non_gravitational','derived_from_g','unknown'].includes(source))throw Error('Massenherkunft ungültig');
 if(!['documented','unverified','nonvacuum','unknown'].includes(vac))throw Error('Vakuumstatus ungültig');
 const out={reference_um:'10.000000',K_IR:null,q:null,delta_lambda_reference_m:null,lambda_B_reference_m:null,c_measured:null,r_s_m:null,GM:null,G_conditional:null,sigma_G_delta_only:null,residual_vs_reference:null,vacuum_status:vac,independent_G_proven:false,status:statuses};
 statuses.push(vac==='documented'?'Vakuum dokumentiert (Angabe, nicht durch Programm geprüft)':'Vakuumstatus offen oder Nichtvakuum; Ursache nicht identifiziert');
 if(delta!==null){
  const K=q(1).add(delta);if(!K.isPositive())throw Error('K_IR muss positiv sein');
  out.K_IR=K.toFixed(36);out.q=q(1).div(K).toFixed(36);
  out.delta_lambda_reference_m=ref.mul(delta).toScientific();out.lambda_B_reference_m=ref.mul(K).toFixed(36);
 }else statuses.push('Einzelmessung delta_k fehlt');
 const c=(lambdaC&&frequencyC)?lambdaC.mul(frequencyC):null;
 if(c){out.c_measured=c.toScientific();statuses.push('c aus externen λ/f-Messpaar berechnet; unabhängige Kalibrierung nachweisen');}
 else statuses.push('c-Messpaar fehlt; kein GM oder G');
 let rs=null;
 if(delta&&rA&&rB){
  if(rB.sub(rA).n<=0n)statuses.push('Nur r_B > r_A und definierter statischer Außenraum unterstützt');
  else{
   rs=invert(delta,rA,rB);
   if(rs.n<0n||rs.sub(rA).n>=0n){rs=null;statuses.push('r_s verletzt Außenfeld-Bedingungen');}
   else{out.r_s_m=rs.toScientific();if(sigma)out.sigma_rs_delta_only_m=slope(delta,rA,rB).abs().mul(sigma).toScientific();}
  }
 }else statuses.push('Radien/Phase fehlen; r_s offen');
 if(rs&&c){
  const gm=c.mul(c).mul(rs).div(2);out.GM=gm.toScientific();
  if(mass&&source==='non_gravitational'){
   const g=gm.div(mass);out.G_conditional=g.toScientific();
   if(sigma)out.sigma_G_delta_only=slope(delta,rA,rB).abs().mul(sigma).mul(c.mul(c)).div(q(2).mul(mass)).toScientific();
   if(gRef&&gRef.isPositive())out.residual_vs_reference=g.div(gRef).sub(1).toScientific();
   statuses.push('G BEDINGT: M deklariert unabhängig; diese Deklaration ist kein Messbeweis');
  }else statuses.push('G offen: keine unabhängig nichtgravitative Masse');
 }else statuses.push('G offen: gültiges r_s oder c fehlt');
 if(gRef)statuses.push('Vergleichs-G wird nicht in Inversion eingespeist');
 statuses.push('Nur delta-Fehler berechnet; c/M/R/Geometrie-Systematik fehlen');
 return out;
}
const api={Q, evaluate, invert};
if(typeof module!=='undefined'&&module.exports)module.exports=api;
root.CIQNoFit=api;
})(typeof globalThis==='object'?globalThis:this);
