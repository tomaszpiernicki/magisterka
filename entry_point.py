import threading

from gen import Generator

i = 0;

def threadFor(start, stop):
    for item in range(start, stop):
        Generator.generate_random_melody()
        print(item)

if __name__ == '__main__':
    for item in range(0, 56):
        Generator.generate_random_melody(item)
        print(item)

    # for n in range(0, 500, 50):
    #     stop = n + 100 if n + 100 <= 1000 else 1000
    #     threading.Thread(target=threadFor, args=(n, stop)).start())