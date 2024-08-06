import requests
import re

o = 1
comment_txt=[]
for page in range(1,11):
    response = requests.get('https://tieba.baidu.com/p/9112492694?pn={}'.format(page), headers={
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36'})

    comment = re.findall('style="display:;">                    (.*?)</div>', response.text)
    name = re.findall('<a data-field=.*?target="_blank">(.*?)</a>',
                      response.text)
    #print(comment)



    for i in range(len(comment)):

        if '<img' in comment[i]:

            url = re.findall('src="(.*?)"',comment[i])[0]
            with open('C:/Users/DELL/Desktop/贴吧图片/图片{}.jpg'.format(o),'wb') as f:
                f.write(requests.get(url).content)
                f.close()
            x = re.sub('<img.*?>', " 图片{}".format(o ), comment[i])
            o+=1
        else:x = comment[i]
        y = re.sub('<img.*?>',"",name[i])

        comment_txt.append(y+" : "+x)

with open('C:/Users/DELL/Desktop/comments.txt', 'w', encoding='utf-8') as file:
    for comment in comment_txt:
        file.write(comment + '\n')

