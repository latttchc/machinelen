import MeCab as mc

t = mc.Tagger("-Owakati")
# 単語ごとに分割された分かち書きの形に変換され、wakati.txt として結果が保存
f = open("data/neko.txt","r", encoding="utf-8")
text = f.readlines() #1行ずつ読み取り
num = len(text) #行数の取得

for i in range(num):
    print(text)
    #形態素解析
    res = t.parse(text[i])
    print(res)
    f2 = open("wakati.txt", "a")
    f2.write(res)
    f2.close()
f.close()