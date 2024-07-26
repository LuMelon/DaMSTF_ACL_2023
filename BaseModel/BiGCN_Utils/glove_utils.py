from tqdm import tqdm, trange
import argparse, pickle
import multiprocessing
import torch

def line_parse(line, queue):
    print("line parse")
    s = line[:-1].split(' ')
    term = s[0]
    values = [float(v) for v in s[1:]]
    # queue.put((term, values))
    return term, torch.tensor(values)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--glove_file', default='./glove.txt', help='filename of the glove file.')
    args = parser.parse_args()
    fname = args.glove_file

    num_processes = multiprocessing.cpu_count()
    print("num_processes : ", num_processes)
    pool = multiprocessing.Pool()
    sh_queue = multiprocessing.Queue()

    cnt= 0
    rst = []
    with open(fname, 'r') as fr:
        for tline in tqdm(fr):
            pool.apply_async(line_parse, args=(tline, sh_queue))
            cnt += 1
            if cnt % 100 == 0:
                for _ in range(100):
                    rst.append(sh_queue.get())
                pool.join()

    pool.close()
    sh_queue.close()

    with open("./tmp.pkl", 'w') as fw:
        pickle.dump(rst, fw)