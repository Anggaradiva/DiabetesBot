import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, MessageHandler, filters, CallbackContext
import logging

# Set up logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

# Muat dataset dan abaikan kolom indeks jika ada
df = pd.read_csv('diabetes.csv', index_col=0)  # Ganti dengan path dataset Anda

# Pisahkan fitur dan label
x = df.drop(columns=['Hasil'])
y = df['Hasil']

# Normalisasi fitur
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)

# Pecah dataset menjadi data latih dan data uji
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Latih model dengan SVM
clf = SVC(kernel='linear', probability=True)
clf.fit(x_train, y_train)

# Prediksi dan hitung akurasi
y_pred = clf.predict(x_test)
CLF_acc = accuracy_score(y_pred, y_test)
classification_report_dict = classification_report(y_test, y_pred, output_dict=True)

# Buat string format laporan klasifikasi secara manual
classification_report_str = (
    "Laporan Klasifikasi:\n\n"
    "Kelas 0 (Negatif Diabetes):\n"
    f"Precision: {classification_report_dict['0']['precision']:.2f}\n"
    f"Recall: {classification_report_dict['0']['recall']:.2f}\n"
    f"F1-Score: {classification_report_dict['0']['f1-score']:.2f}\n"
    f"Support: {classification_report_dict['0']['support']}\n\n"
    "Kelas 1 (Positif Diabetes):\n"
    f"Precision: {classification_report_dict['1']['precision']:.2f}\n"
    f"Recall: {classification_report_dict['1']['recall']:.2f}\n"
    f"F1-Score: {classification_report_dict['1']['f1-score']:.2f}\n"
    f"Support: {classification_report_dict['1']['support']}\n\n"
    "Rata-rata Makro:\n"
    f"Precision: {classification_report_dict['macro avg']['precision']:.2f}\n"
    f"Recall: {classification_report_dict['macro avg']['recall']:.2f}\n"
    f"F1-Score: {classification_report_dict['macro avg']['f1-score']:.2f}\n\n"
    "Rata-rata Tertimbang:\n"
    f"Precision: {classification_report_dict['weighted avg']['precision']:.2f}\n"
    f"Recall: {classification_report_dict['weighted avg']['recall']:.2f}\n"
    f"F1-Score: {classification_report_dict['weighted avg']['f1-score']:.2f}\n\n"
    f"Akurasi Model: {CLF_acc:.2f} (atau {CLF_acc * 100:.2f}%)"
)

logging.info(f"Model accuracy: {CLF_acc:.2f}")

# Plot grafik korelasi
plt.figure(figsize=(10, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.savefig('correlation_matrix.png')
plt.close()

# Inisialisasi SHAP
explainer = shap.Explainer(clf, x_train)
shap_values = explainer(x_test)

# Fungsi untuk memulai bot
async def start(update: Update, context: CallbackContext) -> None:
    context.user_data.clear()
    context.user_data['input_data'] = []
    
    keyboard = [
        [InlineKeyboardButton("Mulai Input Data", callback_data='start_input')],
        [InlineKeyboardButton("Tampilkan Akurasi", callback_data='accuracy')],
        [InlineKeyboardButton("Tampilkan Grafik Korelasi", callback_data='correlation')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        'Halo! Klik tombol di bawah untuk mulai memasukkan data pasien.',
        reply_markup=reply_markup
    )

# Fungsi untuk menangani klik tombol inline
async def button(update: Update, context: CallbackContext) -> None:
    query = update.callback_query
    await query.answer()

    if query.data == 'start_input':
        context.user_data['input_step'] = 1
        await query.edit_message_text('Masukkan jumlah kehamilan:')
    elif query.data == 'accuracy':
        keyboard = [
            [InlineKeyboardButton("Tampilkan Akurasi", callback_data='accuracy')],
            [InlineKeyboardButton("Tampilkan Grafik Korelasi", callback_data='correlation')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            text=f'**Akurasi model saat ini:** {CLF_acc:.2f} (atau {CLF_acc * 100:.2f}%)\n\n'
                 f'**Laporan Klasifikasi:**\n\n'
                 f'{classification_report_str}',
            parse_mode='Markdown',
            reply_markup=reply_markup
        )
    elif query.data == 'correlation':
        with open('correlation_matrix.png', 'rb') as img:
            await query.message.reply_photo(photo=img)
        await query.edit_message_text(
            text='Berikut adalah grafik korelasi.'
        )

# Fungsi untuk menangani pesan dan melakukan klasifikasi
async def classify(update: Update, context: CallbackContext) -> None:
    user_data = context.user_data['input_data']
    user_text = update.message.text
    current_step = context.user_data.get('input_step', 0)
    
    try:
        user_data.append(float(user_text))
        context.user_data['input_step'] += 1

        if current_step == 1:
            await update.message.reply_text('Masukkan kadar glukosa:')
        elif current_step == 2:
            await update.message.reply_text('Masukkan tekanan darah diastolik:')
        elif current_step == 3:
            await update.message.reply_text('Masukkan ketebalan kulit:')
        elif current_step == 4:
            await update.message.reply_text('Masukkan kadar insulin:')
        elif current_step == 5:
            await update.message.reply_text('Masukkan BMI:')
        elif current_step == 6:
            await update.message.reply_text('Masukkan DiabetesPedigreeFunction:')
        elif current_step == 7:
            await update.message.reply_text('Masukkan umur:')
        elif current_step == 8:
            input_df = pd.DataFrame([user_data], columns=[
                'Kehamilan', 'Glukosa', 'Tekanan Darah', 'Ketebalan Kulit',
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Umur'
            ])
            
            input_scaled = scaler.transform(input_df)
            prediction = clf.predict(input_scaled)
            result = 'Positif Diabetes' if prediction[0] == 1 else 'Negatif Diabetes'
            
            await update.message.reply_text(f'Prediksi: {result}')
            
            # Menghitung nilai SHAP untuk input pengguna
            try:
                logging.info(f"Generating SHAP values for input data: {user_data}")
                shap_value = explainer(input_scaled)
                logging.info(f"SHAP values generated successfully")
                shap.initjs()
                shap.force_plot(explainer.expected_value, shap_value.values, input_df, show=False, matplotlib=True)
                plt.savefig('shap_force_plot.png')
                plt.close()
                
                # Menambahkan penjelasan pada grafik SHAP
                explanation = (
                    f'Grafik di atas menunjukkan kontribusi masing-masing fitur terhadap prediksi model.\n\n'
                    f'Fitur yang berkontribusi positif (merah) mendorong model untuk memprediksi "Positif Diabetes", '
                    f'sementara fitur yang berkontribusi negatif (biru) mendorong model untuk memprediksi "Negatif Diabetes".\n\n'
                    f'Berikut adalah penjelasan masing-masing fitur:\n'
                    f'1. **Kehamilan**: Jumlah kehamilan yang pernah dialami.\n'
                    f'2. **Glukosa**: Tingkat glukosa dalam darah.\n'
                    f'3. **Tekanan Darah**: Tekanan darah diastolik (mm Hg).\n'
                    f'4. **Ketebalan Kulit**: Ketebalan lipatan kulit triceps (mm).\n'
                    f'5. **Insulin**: Tingkat insulin serum (mu U/ml).\n'
                    f'6. **BMI**: Indeks Massa Tubuh (berat dalam kg/(tinggi dalam m)^2).\n'
                    f'7. **DiabetesPedigreeFunction**: Fungsi garis keturunan diabetes.\n'
                    f'8. **Umur**: Usia pasien.\n\n'
                    f'Fitur dengan nilai yang lebih tinggi atau lebih rendah dari rata-rata memiliki pengaruh yang lebih kuat terhadap prediksi.'
                )
                
                with open('shap_force_plot.png', 'rb') as img:
                    await update.message.reply_photo(photo=img, caption=explanation)
            except Exception as e:
                logging.error(f"Error during SHAP value calculation: {e}")
                await update.message.reply_text('Terjadi kesalahan saat menghitung nilai SHAP. Silakan coba lagi atau hubungi admin.')
            
            context.user_data.clear()
            await update.message.reply_text('Untuk memulai kembali, ketik /start')
    
    except ValueError as ve:
        logging.error(f"ValueError: {ve}")
        await update.message.reply_text('Format data salah. Harap masukkan nilai numerik yang valid.')
    except Exception as e:
        logging.error(f"Error during classification: {e}")
        await update.message.reply_text('Terjadi kesalahan saat memproses data. Silakan coba lagi atau hubungi admin jika masalah berlanjut.')

# Fungsi utama untuk menjalankan bot
def main() -> None:
    # Masukan Token
    application = Application.builder().token('Token_bot').build()
    
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CallbackQueryHandler(button))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, classify))
    
    application.run_polling()

if __name__ == '__main__':
    main()
