import urllib
import requests
import multiprocessing.pool
from multiprocessing import Pool
import uuid
import os

images_dir = os.path.join("data", "train")
small_letters = map(chr, range(ord('a'), ord('f')+1))
digits = map(chr, range(ord('0'), ord('9')+1))

base_16 = digits + small_letters

MAX_THREADS = 100

def captcha(code):
    try:
        r = requests.get("https://local.thedrhax.pw/rucaptcha/?" + code)
        filename = code + "_" + str(uuid.uuid1().time) + ".png"
        path = os.path.join(images_dir, filename)
        with open(path, "wb") as png:
            png.write(bytes(r.content))
            print("Downloaded " + str(code))
    except Exception as e:
        print(str(e))

if __name__ == "__main__":

    labels = []
    for i in range(0, len(base_16)):
        for j in range(0, len(base_16)):
            for m in range(0, len(base_16)):
                for n in range(0, len(base_16)):
                    try:
                        label = base_16[i] + base_16[j] + base_16[m] + base_16[n]
                        labels.append(label)
                        # urllib.urlretrieve("https://local.thedrhax.pw/rucaptcha/?" + str(label), str(label) + ".png")                        
                    except Exception as e:
                        print(str(e))

    print(labels)

    p = Pool(MAX_THREADS)
    while 1:
        p.map(captcha, labels)

    print("Finished all downloads")