from LongCLIP.module import LongCLIP
import torch
import os
import json
import base64

Datasetpath = '/app/Dataset'
ImgEmbedname = 'imgembed.json'

def processing(text) :
    clip = LongCLIP()
    tf, _ = clip.run(text)
    jsondata = getJsonData()
    imf = getImgEmbed(jsondata)
    sim = clip.get_CosSim(tf,imf)
    val,indi = torch.topk(sim,5)
    results = []
    for i in indi :
        id = jsondata[i]['id']
        with open(os.path.join(Datasetpath,f'{id}.png'), 'rb') as imf: 
            encoded_string = base64.b64encode(imf.read()).decode('utf-8')
        data = {'id' : id, 'url' : jsondata[i]['url'], 'img' : encoded_string}
        results.append(data)
    result = {'data' : results}
    return result

def getImgEmbed(data) :
    embedvecs = []
    for d in data :
        embedvecs.append(d['embed'])
    
    return torch.tensor(embedvecs)

def getJsonData() :
    with open(os.path.join(Datasetpath,ImgEmbedname),'r') as jf :
        data = json.load(jf)
    return data

# if __name__ == '__main__' :
#     result = processing('hello nice meet you')
#     with open('/app/test.json','w',encoding='utf-8') as jf :
#         json.dump(result,jf,ensure_ascii=False,indent=4)
    
#     with open('/app/test.json','rb') as jf :
#         data = json.load(jf)
#     imgdata = base64.b64decode(data['data'][0]['img'])
#     with open('test.png','wb') as imffile:
#         imffile.write(imgdata)
