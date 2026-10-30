"""Read the original ZIP without rotating or rewriting source JPEGs."""
import zipfile,math,hashlib
from pathlib import PurePosixPath
import cv2,numpy as np

class VisDroneZIP:
 def __init__(self,path,expected=548):
  self.archive=zipfile.ZipFile(path)
  names=self.archive.namelist()
  if len(names)!=len(set(names)):raise ValueError('Duplicate ZIP members')
  self.images=sorted(n for n in names if '/images/' in n and n.lower().endswith(('.jpg','.jpeg','.png')) and not PurePosixPath(n).name.startswith('._') and '__MACOSX' not in n)
  self.labels={PurePosixPath(n).stem:n for n in names if '/annotations/' in n and n.endswith('.txt') and not PurePosixPath(n).name.startswith('._') and '__MACOSX' not in n}
  if len(self.images)!=expected or {PurePosixPath(n).stem for n in self.images}!=set(self.labels):raise ValueError('Full validation membership mismatch')
 def record(self,index):
  name=self.images[index];stem=PurePosixPath(name).stem;image_bytes=self.archive.read(name);label_bytes=self.archive.read(self.labels[stem])
  image=cv2.imdecode(np.frombuffer(image_bytes,np.uint8),cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)
  if image is None:raise ValueError('Decode failed:'+name)
  height,width=image.shape[:2];objects=[];ignores=[];others=[]
  for row,line in enumerate(label_bytes.decode('utf-8-sig').splitlines(),1):
   if not line.strip():continue
   v=[float(x) for x in line.split(',') if x.strip()]
   if len(v)<8 or not all(math.isfinite(x) for x in v):raise ValueError('Malformed annotation')
   x,y,w,h,score,cat=v[:6]
   if w<=0 or h<=0 or cat!=int(cat) or not 0<=cat<=11:raise ValueError('Invalid annotation')
   x1,y1=max(0,x),max(0,y);x2,y2=min(width,x+w),min(height,y+h)
   if x2<=x1 or y2<=y1:raise ValueError('Empty clipped annotation')
   box=[x1,y1,x2-x1,y2-y1]
   if cat==0 or score==0:ignores.append(box)
   elif cat==11:others.append(box)
   else:objects.append(dict(class_id=int(cat)-1,bbox=box))
  rec=dict(id=index+1,stem=stem,file_name=PurePosixPath(name).name,width=width,height=height,objects=objects,ignore_regions=ignores,other_regions=others)
  provenance=dict(image_id=index+1,ZIP_member=name,image_SHA256=hashlib.sha256(image_bytes).hexdigest(),annotation_SHA256=hashlib.sha256(label_bytes).hexdigest(),decode_flag='IMREAD_COLOR|IMREAD_IGNORE_ORIENTATION',shape=list(image.shape))
  return image,rec,provenance
 def close(self):self.archive.close()
