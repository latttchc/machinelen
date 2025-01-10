f = open("wakati.txt", "r")

word2id = {}
id2word = {}

text = f.readlines()
num = len(text)
id = 0 
for i in range(num):
    for word in text[i].split(" "): #split:List内の要素を分割
        if word == "\n":
            continue
        else:
            if word not in word2id:
                id = id + 1
                word2id[word] = id #word to id
                id2word[id] = word #id to word

f.close()
print("word2id:",word2id)
print("id2word:",id2word)