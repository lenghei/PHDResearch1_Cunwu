"""Four-corner algebra; a value is unavailable unless all corners exist."""
def did(corners):
 if set(corners)!={'Normal','Motion','Haze','MotionHaze'} or any(v is None for v in corners.values()):return None
 return corners['MotionHaze']-corners['Motion']-corners['Haze']+corners['Normal']
