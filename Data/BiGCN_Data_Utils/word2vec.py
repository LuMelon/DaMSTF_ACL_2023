from Dataloader.twitterloader import *
from gensim.models import word2vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def plot_w2v(emb):
    tsne = TSNE(metric='cosine', random_state=123)
    embed_tsne = tsne.fit_transform(emb)
    fig, ax = plt.subplots(figsize=(14, 14))

    for idx in range(len(emb)):
        plt.scatter(*embed_tsne[idx, :], color='steelblue')
    plt.show()

if __name__ == '__main__':
    dir = "../../../pheme-rnr-dataset"
    events = [os.path.join(dir, item) for item in os.listdir(dir)]
    events = [e for e in events if os.path.isdir(e)]
    te = TwitterSet()
    te.load_event_list(events, cached_pkl_file='../data/pheme.pkl')

    texts = [text for ID in te.data_ID for text in te.data[ID]['text']]
    model = word2vec.Word2Vec(texts, size=100, window=2, min_count=1, workers=4)

    with open("./w2v.pkl", "wb") as fw:
        pickle.dump(model, fw, protocol=pickle.HIGHEST_PROTOCOL)
