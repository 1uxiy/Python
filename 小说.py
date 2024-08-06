import requests
import re
from collections import OrderedDict


novel = OrderedDict()
num_chapter = 20
url = 'https://m.biquge.asia/book/57550/16917176.html'
for i in range(0,num_chapter):
    response = requests.get(url=url,
                            headers={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36'})
    response.encoding = 'GBK'

    text_list = re.findall('<div id="nr1">&nbsp;&nbsp;&nbsp;&nbsp;(.*?)<br/></div>', response.text)
    text = re.sub('<br/><br/>&nbsp;&nbsp;&nbsp;&nbsp;','\n',text_list[0])

    chapter_list = re.findall('<div class="nr_title" id="chaptertitle">(.*?)</div>', response.text)
    novel.setdefault(chapter_list[0],text)


    url_list = re.findall('<a href="(.*?)">下一章</a>', response.text)
    url = 'https://m.biquge.asia'+url_list[0]


for key in novel:
    with open('C:/Users/DELL/Desktop/novel/{}.txt'.format(key),'w')as f:
        f.write(novel[key])
        f.close()
