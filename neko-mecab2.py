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

def w2id(word2id, length):
    result = []
    f = open("wakati.txt", "r")
    text = f.readlines()
    for i in range(len(text)):
        tmp = []
        for word in text[i].split(" "):
            if word in word2id:
                tmp.append(word2id[word])
            if word == "\n":
                num = len(tmp)
                if num < length:
                    for i in range(length - num):
                        tmp.append(0)
                result.append(tmp)
    f.close()
    return result

id_list = w2id(word2id, 30)
print(id_list)