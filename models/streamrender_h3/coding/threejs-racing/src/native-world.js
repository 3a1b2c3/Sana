import * as T from 'three';
import {loadNative} from './native-assets.js';
export async function makeWorld(scene){
 const [track,data]=await Promise.all([loadNative('track'),fetch('/torcs/centerline.json').then(r=>r.json())]);scene.add(track);
 const curve=new T.CatmullRomCurve3(data.points.map(p=>new T.Vector3(...p)),true,'centripetal');curve.arcLengthDivisions=8000;curve.updateArcLengths();
 const length=curve.getLength(),samples=Array.from({length:801},(_,i)=>{const point=curve.getPointAt(i/800),tangent=curve.getTangentAt(i/800);return{point,tangent,normal:new T.Vector3(-tangent.z,0,tangent.x).normalize()}});
 function nearest(position){let best=samples[0],dist=Infinity,index=0;for(let i=0;i<800;i++){const d=(position.x-samples[i].point.x)**2+(position.z-samples[i].point.z)**2;if(d<dist){dist=d;best=samples[i];index=i}}return{...best,index,distance:Math.sqrt(dist),lateral:position.clone().sub(best.point).dot(best.normal)}}
 // Sample actual road triangles for height and banking; no synthetic terrain approximation.
 const roads=track.children.filter(m=>m.userData.semantic==='road'),ray=new T.Raycaster();track.updateMatrixWorld(true);
 function heightAt(x,z){ray.set(new T.Vector3(x,150,z),new T.Vector3(0,-1,0));const hit=ray.intersectObjects(roads,false)[0];if(hit)return hit.point.y+.08;const n=nearest(new T.Vector3(x,0,z));return n.point.y+.08;}
 const approx=new T.Vector3(1089.463,35.262,-574.777),near=nearest(approx),startFraction=near.index/800,startDistance=startFraction*length,start=curve.getPointAt(startFraction),direction=curve.getTangentAt(startFraction);
 // A traffic cone just ahead of the spawn point, opt-in via ?cones=1. Labeled
 // 'barrier' (not a new semantic class) so it stays in the H3 model's known
 // segmentation vocabulary -- orange in the palette either way.
 // ?cones=<N> places N cones down the middle of the road, spaced out, starting
 // well down the road from spawn; omitted or 0 means none.
 const coneCount=Number(new URLSearchParams(location.search).get("cones")||0);
 console.warn('[cones] count',coneCount);
 if(coneCount>0){
  // Follow the actual curve (arc-length offset from spawn), not a straight
  // line projected from the spawn tangent -- a straight line drifts off the
  // road as soon as the track bends.
  const bodyHeight=.7;
  Array.from({length:coneCount},(_,i)=>i).forEach(i=>{
   const t=((startDistance+25+i*2)%length)/length,base=curve.getPointAt(t);
   base.y=heightAt(base.x,base.z);
   const plateHeight=.05;
   const plate=new T.Mesh(new T.BoxGeometry(.4,plateHeight,.4),new T.MeshStandardMaterial({color:0xff6a13,roughness:.86}));
   plate.position.set(base.x,base.y+plateHeight/2,base.z);plate.userData.semantic='barrier';plate.castShadow=true;plate.receiveShadow=true;
   scene.add(plate);
   const cone=new T.Mesh(new T.ConeGeometry(.25,bodyHeight,12),new T.MeshStandardMaterial({color:0xff6a13,roughness:.86}));
   cone.position.set(base.x,base.y+plateHeight+bodyHeight/2,base.z);cone.userData.semantic='barrier';cone.castShadow=true;cone.receiveShadow=true;
   scene.add(cone);
   const band=new T.Mesh(new T.CylinderGeometry(.12,.135,.05,12),new T.MeshStandardMaterial({color:0xf2f2f2,roughness:.86}));
   band.position.set(base.x,base.y+plateHeight+bodyHeight*.62,base.z);band.userData.semantic='barrier';band.receiveShadow=true;
   scene.add(band);
  });
 }
 return{roadHalfWidth:6,curve,length,samples,nearest,updateTrees:()=>{},heightAt,bounds:{radius:2000},startDistance,spawn:{x:start.x,z:start.z,heading:Math.atan2(-direction.x,-direction.z)}};
}
