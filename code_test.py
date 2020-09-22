from roc import *

TOTAL_CSV_DIR = 'E:/data/viennacode_img_19600101_20191231_unique_preprocessed/labels_small.csv'
TEST_CSV_DIR = 'E:/data/viennacode_img_19600101_20191231_unique_preprocessed/test_labels_small.csv'
IMG_DIR = 'E:/data/viennacode_img_19600101_20191231_unique_preprocessed/imgs'
MODEL_DIR = './TRADEMARK_SMALL_AG_CNN'
SAVE_DIR = './TRADEMARK_SMALL_AG_CNN_ROC'
DISCRIMINATOR = '|'
BATCH_SIZE = 128

get_model_outputs = getModelOutputs(total_csv_dir=TOTAL_CSV_DIR,
                                    test_csv_dir=TEST_CSV_DIR,
                                    img_dir=IMG_DIR,
                                    model_dir=MODEL_DIR,
                                    save_dir=SAVE_DIR,
                                    discriminator=DISCRIMINATOR,
                                    batch_size=BATCH_SIZE)

if not os.path.exists(get_model_outputs.g_save_path):
    get_model_outputs.getOutputs()
    onehot_labels = get_model_outputs.onehot_label_df
    g_results = get_model_outputs.g_df
    l_results = get_model_outputs.l_df
    f_results = get_model_outputs.f_df
else:
    get_model_outputs.saveHeatMaps()
    onehot_labels = pd.read_csv(get_model_outputs.label_save_path)
    g_results = pd.read_csv(get_model_outputs.g_save_path)
    l_results = pd.read_csv(get_model_outputs.l_save_path)
    f_results = pd.read_csv(get_model_outputs.f_save_path)

evaluation = Evaluation(onehot_df=onehot_labels,
                        g_df=g_results,
                        l_df=l_results,
                        f_df=f_results,
                        mlb=get_model_outputs.MLB,
                        save_dir=SAVE_DIR)
evaluation.execute()