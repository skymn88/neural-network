"""New independent excitation seeds, same clean physical plant as v6."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import savemat
from scipy.signal import resample_poly
ROOT=Path(__file__).resolve().parents[2]/"data"/"synthetic"
from scipy.signal import lfilter
import json
SEED=20261110
DUR=[500,550,600,650,700,750,800,900,1000,1100,850]
TRAIN=[0,1,3,4,6,7,9];VAL=[2,5];TEST=[8,10]
PROFILES=['long_hold','long_hold','hold_validation','long_hold','transitions','transition_validation','long_hold','transitions','hold_test','transitions','transition_test']

def piecewise(rng,t,lo,hi,amp):
 y=np.empty(len(t));k=0
 while k<len(t):
  count=max(1,int(rng.uniform(lo,hi)*256));end=min(len(t),k+count)
  y[k:end]=rng.uniform(-amp,amp);k=end
 return y

def generate_rich():
 flights=[];stats=[]
 for f,duration in enumerate(DUR):
  rng=np.random.default_rng(SEED+1009*f);dt=1/256;n=duration*256+1;t=np.arange(n)*dt
  profile=PROFILES[f];calm='hold' in profile;burst='transition' in profile
  target=piecewise(rng,t,45 if calm else 20,160 if calm else 100,.05)
  target[:int(10*256)]=-.045;target[int(10*256):int(18*256)]=0
  # Slewed reference alternates smooth ramps with faster requests.
  tau=.08 if burst else .3;a=np.exp(-dt/tau);target=lfilter([1-a],[1,-a],target,zi=[a*target[0]])[0]
  pulse=piecewise(rng,t,.12,.9,1.4 if burst else .6)
  noise=lfilter([1-np.exp(-dt/.07)],[1,-np.exp(-dt/.07)],rng.normal(size=n))
  # Quiet / active alternation; no injected plant oscillation or motor vibration.
  changes=np.r_[0,np.flatnonzero(abs(np.diff(target))>.00005)]
  # Detect starts of command transitions, then irregular decaying input bursts.
  onset=changes[np.r_[True,np.diff(changes)>256]]
  gate=np.zeros(n,dtype=bool)
  for at in onset:gate[at:min(n,at+int(rng.uniform(1,5)*256))]=True
  excite=(pulse+.45*noise)*gate*(.15 if calm else .5)
  if 'small' in profile:
   target*=.25;excite=.14*np.sign(pulse)+.04*noise
  z=zd=v=0.;x=-.045+.0005*f;current=0.;arr=np.empty((n,7));stops=0
  for k in range(n):
   request=np.clip(85*(target[k]-x)-12*v+excite[k],-4,4)
   current+=np.clip(request-current,-25*dt,25*dt)
   drive=np.tanh(np.sign(current)*max(abs(current)-.04,0)/2.3)
   zd+=dt*((2*np.pi*12)**2*(drive-z)-2*.7*(2*np.pi*12)*zd);z+=dt*zd
   flow=np.sign(z)*max(abs(z)-.025,0)
   veq=.075*flow-.025*x
   acc=(veq-v)/.055-.012*np.tanh(v/.001)
   v+=dt*acc;x+=dt*v
   if abs(x)>.065:stops+=1;x=np.clip(x,-.065,.065);v=0
   arr[k]=[current,z,x,400*x+150*x**3,18000*x+2200*v+90*acc,21e6,1e6]
  b=resample_poly(arr,1,4,axis=0,padtype='line');tt=np.arange(len(b))/64
  flights.append({'name':f'RICH_{f+1:02d}','t':tt,'a':b})
  stats.append(dict(Flight=f+1,Seed=SEED+1009*f,Profile=profile,Seconds=duration,Split='train' if f in TRAIN else 'validation' if f in VAL else 'test',CurrentMin=b[:,0].min(),CurrentMax=b[:,0].max(),CurrentStd=b[:,0].std(),CurrentSlewRMS=np.sqrt(np.mean((np.diff(b[:,0])*64)**2)),CurrentReversals=int(np.sum(b[1:,0]*b[:-1,0]<0)),RAMMin=b[:,2].min(),RAMMax=b[:,2].max(),NativeStopContacts=stops))
  print('GENERATED',f+1,profile,'stops',stops,flush=True)
 names=['CoilCurrent','DDV','RAM','Deflection','ActuatorForce','SupplyPressure','ReturnPressure'];arr=np.empty((1,len(DUR)),dtype=[('name','O')]+[(s,'O') for s in names])
 for i,fl in enumerate(flights):
  arr['name'][0,i]=fl['name']
  for j,s in enumerate(names):arr[s][0,i]={'t':fl['t'][:,None],'x':fl['a'][:,j,None]}
 savemat(ROOT/'synthetic_rich.mat',dict(Raw=arr,trainIDs=np.array(TRAIN)+1,validationIDs=np.array(VAL)+1,testIDs=np.array(TEST)+1),do_compression=True)
 pd.DataFrame(stats).to_excel(ROOT/'excitation_manifest.xlsx',index=False)
 (ROOT/'generation.json').write_text(json.dumps({'seed':SEED,'profiles':PROFILES,'plant':'same clean v6 plant','slewLimitAperSecond':25,'sampleRate':64,'nativeSimulationRate':256,'measurementNoise':False,'motorVibration':False},indent=2))
 return flights
if __name__=='__main__':generate_rich()
