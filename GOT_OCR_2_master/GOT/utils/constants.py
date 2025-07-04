CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15
tokenized = True
LOGDIR = "log"

IGNORE_INDEX = -100
# DEFAULT_PAD_TOKEN = "[PAD]"

DEFAULT_PAD_TOKEN = "<|endoftext|>"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "<unk>"
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_BOX_TOKEN = "<box>"

DEFAULT_IMAGE_PATCH_TOKEN = '<imgpad>'

DEFAULT_IM_START_TOKEN = '<img>'
DEFAULT_IM_END_TOKEN = '</img>'

if tokenized:
    CONVERSATION_DATA = {

        'data_1': {
            'images': '/data_8t_1/dataset/tfr-dataset/image_and_json/TR/images',
            'annotations': '/data_8t_1/dataset/tfr-dataset/image_and_json/TR/clean_data_tokenized.json',
        },
        'data_2': {
            'images': '/data_8t_1/dataset/tfr-dataset/image_and_json/TFR/images',
            'annotations': '/data_8t_1/dataset/tfr-dataset/image_and_json/TFR/clean_data_tokenized.json',
        },
        'data_3': {
            'images': '/data_8t_1/dataset/tfr-dataset/image_and_json/MFR/images',
            'annotations': '/data_8t_1/dataset/tfr-dataset/image_and_json/MFR/clean_data_tokenized.json',
        },
        'data_4': {
            'images': '/data_8t_1/dataset/tfr-dataset/MFR/val_image_and_json/images',
            'annotations': '/data_8t_1/dataset/tfr-dataset/MFR/val_image_and_json/data_tokenized.json',
        },
        'data_5': {
            'images': '/apps/GOT-OCR2.0/test_image/output_images',
            'annotations': '/apps/GOT-OCR2.0/test_image/longest_gpt_replies_full_tokenized.json',
        }

    }

else:
    CONVERSATION_DATA = {

        'data_1': {
            'images': '/data_8t_1/dataset/tfr-dataset/image_and_json/TR/images',
            'annotations': '/data_8t_1/dataset/tfr-dataset/image_and_json/TR/clean_data.json',
        },
        'data_2': {
            'images': '/data_8t_1/dataset/tfr-dataset/image_and_json/TFR/images',
            'annotations': '/data_8t_1/dataset/tfr-dataset/image_and_json/TFR/clean_data.json',
        },
        'data_3': {
            'images': '/data_8t_1/dataset/tfr-dataset/image_and_json/MFR/images',
            'annotations': '/data_8t_1/dataset/tfr-dataset/image_and_json/MFR/clean_data.json',
        },
        'data_4': {
            'images': '/data_8t_1/dataset/tfr-dataset/MFR/val_image_and_json/images',
            'annotations': '/data_8t_1/dataset/tfr-dataset/MFR/val_image_and_json/data.json',
        },
        'data_5': {
            'images': '/apps/GOT-OCR2.0/test_image/output_images',
            'annotations': '/apps/GOT-OCR2.0/test_image/longest_gpt_replies_full.json',
        }

    }
