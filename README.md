# BIV-hackathon

Решение кейса 1 команды 

## Как запустить проект
'''
docker pull emshen6/biv-case1-copium-solution
docker run --rm emshen6/biv-case1-copium-solution

'''

Получить метки классов на хост

'''
docker ps -a
docker cp <id_контейнера>:/copium/submission.tsv ./
'''

Модель ruBert-tiny finetunned: ['Google Drive'](https://drive.google.com/file/d/1gWDHP381W1Sy-TOTLphdap029gsQHyO3)