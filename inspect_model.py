import pickle
import numpy as np

FILE_PATH = 'microgpt_model_cache.pkl'

def inspect_pickle_model():
    print(f"در حال بارگذاری فایل {FILE_PATH}...\n")
    try:
        with open(FILE_PATH, 'rb') as f:
            model_data = pickle.load(f)
    except FileNotFoundError:
        print("خطا: فایل پیدا نشد!")
        return

    # نمایش پیکربندی مدل (Architecture Config)
    if 'cfg' in model_data:
        print("⚙️ تنظیمات معماری مدل (Model Config):")
        for k, v in model_data['cfg'].items():
            print(f"  - {k:<15}: {v}")
        print("-" * 75)

    # نمایش تنظیمات زمان آموزش (Training Config)
    if 'train_cfg' in model_data:
        print("🎓 تنظیمات آموزش (Training Config):")
        for k, v in model_data['train_cfg'].items():
            print(f"  - {k:<15}: {v}")
        print("-" * 75)

    # استخراج شبکه‌ها از داخل بخش weights
    if 'weights' in model_data:
        weights = model_data['weights']
    else:
        print("خطا: کلید weights در فایل یافت نشد.")
        return

    total_params = 0
    print(f"{'نام لایه / اتصال (Layer Name)':<45} | {'ابعاد (Shape)':<15} | {'تعداد پارامتر'}")
    print("=" * 75)

    for layer_name, weight_matrix in weights.items():
        if isinstance(weight_matrix, np.ndarray):
            shape = weight_matrix.shape
            param_count = weight_matrix.size
            total_params += param_count
            print(f"{layer_name:<45} | {str(shape):<15} | {param_count:,}")
        else:
            print(f"{layer_name:<45} | نوع داده: {type(weight_matrix).__name__}")

    print("=" * 75)
    print(f"✅ مجموع کل پارامترهای آموزش دیده: {total_params:,}")

if __name__ == "__main__":
    inspect_pickle_model()
