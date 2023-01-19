import numpy as np
from lyd import metrics, utils
import copy
import pandas as pd

def deltaFeatures(postData):
    myData = copy.copy(postData)

    for col,data_ in postData.items():
        if "pre" in col:
            pre = col
            post = col.replace('pre','post')
            delta = col.replace('pre','delta')

            if type(postData[col]) is dict:
                a = postData[post]
                b = postData[pre]
                myData[delta] = {key: a[key] - b.get(key, 0) for key in a.keys()}

            else:
                myData[delta] = postData[post]-postData[pre]
                
            #Compute ECR
            if 'dr' in col:
                ecr = col.replace("pre_dr","ecr")
                myData[ecr] = postData[pre]/postData[post]

    return myData

def BSMC(X):
    '''
    Binaural source modulation correlation 
    We measure the correlaiton between the sources of left and right ear. 
    '''
    icBroad, icFb = metrics.ASMC(np.expand_dims(X[0],0),np.expand_dims(X[1],0))
    return icBroad,icFb


def extractFeatures_1(x,xf,xb,prefix = '',fbPipe=None,vmask=None):
    '''
    Extracts features (besides ASMC)
    '''
    
    data = {}
    
    prefix =  prefix + "_"
    X = fbPipe.transform(x)
    Xf = fbPipe.transform(xf)
    Xb = fbPipe.transform(xb)
    
    #v,u = timit.splitVmask(xf[:,0:len(vmask)],vmask)
    #V = fbPipe.transform(v)
    #U = fbPipe.transform(u)
    
    stft_fs = fbPipe.processes[0].fs_stft
    
    # Dynamics
    dr = metrics.DynamicRange(mean=False)
    asmc = metrics.ASMC()
    #ms = spectrum.modulationSpectrum(fs=stft_fs)
    #psc = metrics.Percentiles()
    
    # Dynamic range
    data[prefix+"dr_m"] = dr.transform(x).mean(axis=0)
    data[prefix+"dr_f"] = dr.transform(xf).mean(axis=0)
    data[prefix+"dr_b"] = dr.transform(xb).mean(axis=0)
    
    data[prefix+"dr_fb_m"] = dr.transform(X).mean(axis=0)
    data[prefix+"dr_fb_f"] = dr.transform(Xf).mean(axis=0)
    data[prefix+"dr_fb_b"] = dr.transform(Xb).mean(axis=0)
    
    # Modulation spectrum
    #data[prefix+"ms_fb_m"] = ms.transform(X).mean(axis=0)
    #data[prefix+"ms_fb_f"] = ms.transform(Xf).mean(axis=0)
    #data[prefix+"ms_fb_b"] = ms.transform(Xb).mean(axis=0)
    
    #FBR
    snr = utils.calcSNR(xf,xb)
    snr = np.minimum(30,np.maximum(snr,-30))
    data[prefix + "snr"] = snr.mean(axis=0)
    
    
    #FBR fb
    snr = utils.calcSNR(Xf,Xb)
    snr = np.minimum(30,np.maximum(snr,-30))
    data[prefix +"snr_fb"] = snr.mean(axis=0)
    
    #UVR
    #data[prefix +"uvr"] = utils.calcSNR(u,v).mean(axis=0)
    #data[prefix+"uvr_fb"] = utils.calcSNR(U,V).mean(axis=0)
    
    #level
    data[prefix+"rms_f"] = utils.to_dB(utils.rms(xf).mean(axis=0))
    data[prefix+"rms_b"] = utils.to_dB(utils.rms(xb).mean(axis=0))
    data[prefix+"rms_m"] = utils.to_dB(utils.rms(x).mean(axis=0))
    
    #level fb
    rmsFB = lambda X: np.sqrt(np.mean(X**2,axis=2))
    data[prefix+"rms_fb_f"] = utils.to_dB(rmsFB(Xf).mean(axis=0))
    data[prefix+"rms_fb_b"] = utils.to_dB(rmsFB(Xb).mean(axis=0))
    data[prefix+"rms_fb_m"] = utils.to_dB(rmsFB(X).mean(axis=0))
    
    
    #ASMC
    #asmc, asmc_fb = metrics.ASMC(Xf,Xb,X)
    data[prefix+"asmc"] = asmc(xf,xb).mean()
    data[prefix+"asmc_fb"] = asmc(xf,xb)
    

    #BSMC
    data[prefix+"bsmc_fb_m"] = asmc(x)
    data[prefix+"bsmc_fb_f"] = asmc(xf)
    data[prefix+"bsmc_fb_b"] = asmc(xb)

    #BSMC
    #bsmc_m, bsmc_fb_m = BSMC(X) 
    #data[prefix+"bsmc_m"] = bsmc_m.mean(axis=0)
    #data[prefix+"bsmc_fb_m"] = bsmc_fb_m.mean(axis=0)
    
    #bsmc_f, bsmc_fb_f = BSMC(Xf) 
    #data[prefix+"bsmc_f"] = bsmc_f.mean(axis=0)
    #data[prefix+"bsmc_fb_f"] = bsmc_fb_f.mean(axis=0)
    
    #bsmc_b, bsmc_fb_b = BSMC(Xb) 
    #data[prefix+"bsmc_b"] = bsmc_b.mean(axis=0)
    #data[prefix+"bsmc_fb_b"] = bsmc_fb_b.mean(axis=0)
    

    return data,Xf,Xb

def runAnalysis(scene,comprs,fbPipe):
    # Extract preprocessed signals
    x = scene.mixture
    xf = scene.foreground
    xb = scene.background
    
    df = pd.DataFrame()
    
    snr = scene.SNR
    noiseModDepth = scene.noiseModDepth
    room = scene.brir.room
    
    preData,Xf,Xb = extractFeatures_1(x,xf,xb,prefix='pre',vmask = scene.Vmask,fbPipe=fbPipe) 
    
    preData.update(
    {
        'in_SNR' : snr,
        'in_noiseModDepth' : noiseModDepth,
        'in_room' : room
    })
    
    for key,compr in comprs.items():
        xc = compr.transform(x,background = xb if key in ['aware','ideal'] else None)
        xfc = compr.shadowFilter(xf,target=True)
        xbc = compr.shadowFilter(xb,target=False)

        postData,Xcf,Xcb = extractFeatures_1(xc,xfc,xbc,fbPipe=fbPipe,prefix='post',vmask = scene.Vmask)
        postData.update({'compr' : key})
        
        #Compute FES
        fes_m,fes_fb_m = metrics.FES(Xf+Xb,Xcf+Xcb,Xcf+Xcb)
        postData.update({'fes_m' : fes_m.mean(axis=0),
                        'fes_fb_m':fes_fb_m.mean(axis=0)})
        
        fes_f,fes_fb_f = metrics.FES(Xf,Xcf,Xcf+Xcb)
        postData.update({'fes_f' : fes_f.mean(axis=0),
                        'fes_fb_f':fes_fb_f.mean(axis=0)})
        
        fes_b,fes_fb_b = metrics.FES(Xb,Xcb,Xcf+Xcb)
        postData.update({'fes_b' : fes_b.mean(axis=0),
                        'fes_fb_b':fes_fb_b.mean(axis=0)})
        
        
    
        postData.update(preData)
        
        
        
    
        # Compute delta features
        postData = deltaFeatures(postData)
    
        df = df.append(postData,ignore_index=True)
        
    
    return df