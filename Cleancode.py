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

# Atur logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)


class DiabetesPredictor:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path, index_col=0)
        self.x = self.df.drop(columns=['Hasil'])
        self.y = self.df['Hasil']
        self.scaler = StandardScaler()
        self.x = self.scaler.fit_transform(self.x)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.2, random_state=42)
        self.model = SVC(kernel='linear', probability=True)
        self.model.fit(self.x_train, self.y_train)
        self.y_pred = self.model.predict(self.x_test)
        self.accuracy = accuracy_score(self.y_pred, self.y_test)
        self.classification_report_dict = classification_report(self.y_test, self.y_pred, output_dict=True)
        self.explainer = shap.Explainer(self.model, self.x_train)
        self.correlation_matrix_path = 'correlation_matrix.png'
        self.plot_correlation_matrix()

    def plot_correlation_matrix(self):
        plt.figure(figsize=(10, 8))
        correlation_matrix = self.df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Matrix')
        plt.savefig(self.correlation_matrix_path)
        plt.close()

    def get_classification_report(self):
        report = self.classification_report_dict
        return (
            "Laporan Klasifikasi:\n\n"
            "Kelas 0 (Negatif Diabetes):\n"
            f"Precision: {report['0']['precision']:.2f}\n"
            f"Recall: {report['0']['recall']:.2f}\n"
            f"F1-Score: {report['0']['f1-score']:.2f}\n"
            f"Support: {report['0']['support']}\n\n"
            "Kelas 1 (Positif Diabetes):\n"
            f"Precision: {report['1']['precision']:.2f}\n"
            f"Recall: {report['1']['recall']:.2f}\n"
            f"F1-Score: {report['1']['f1-score']:.2f}\n"
            f"Support: {report['1']['support']}\n\n"
            "Rata-rata Makro:\n"
            f"Precision: {report['macro avg']['precision']:.2f}\n"
            f"Recall: {report['macro avg']['recall']:.2f}\n"
            f"F1-Score: {report['macro avg']['f1-score']:.2f}\n\n"
            "Rata-rata Tertimbang:\n"
            f"Precision: {report['weighted avg']['precision']:.2f}\n"
            f"Recall: {report['weighted avg']['recall']:.2f}\n"
            f"F1-Score: {report['weighted avg']['f1-score']:.2f}\n\n"
            f"Akurasi Model: {self.accuracy:.2f} (atau {self.accuracy * 100:.2f}%)"
        )

    def predict(self, input_data):
        input_scaled = self.scaler.transform([input_data])
        prediction = self.model.predict(input_scaled)
        result = 'Positif Diabetes' if prediction[0] == 1 else 'Negatif Diabetes'
        return result, input_scaled

    def explain_prediction(self, input_scaled, input_data):
        shap_values = self.explainer(input_scaled)
        shap.initjs()
        force_plot_path = 'shap_force_plot.html'
        shap.save_html(force_plot_path, shap.force_plot(self.explainer.expected_value, shap_values.values, pd.DataFrame([input_data])))

        explanation = (
            'Grafik di atas menunjukkan kontribusi masing-masing fitur terhadap prediksi model.\n\n'
            'Fitur yang berkontribusi positif (merah) mendorong model untuk memprediksi "Positif Diabetes", '
            'sementara fitur yang berkontribusi negatif (biru) mendorong model untuk memprediksi "Negatif Diabetes".\n\n'
            'Berikut adalah penjelasan masing-masing fitur:\n'
            '1. **Kehamilan**: Jumlah kehamilan yang pernah dialami.\n'
            '2. **Glukosa**: Tingkat glukosa dalam darah.\n'
            '3. **Tekanan Darah**: Tekanan darah diastolik (mm Hg).\n'
            '4. **Ketebalan Kulit**: Ketebalan lipatan kulit triceps (mm).\n'
            '5. **Insulin**: Tingkat insulin serum (mu U/ml).\n'
            '6. **BMI**: Indeks Massa Tubuh (berat dalam kg/(tinggi dalam m)^2).\n'
            '7. **DiabetesPedigreeFunction**: Fungsi garis keturunan diabetes.\n'
            '8. **Umur**: Usia pasien.\n\n'
            'Fitur dengan nilai yang lebih tinggi atau lebih rendah dari rata-rata memiliki pengaruh yang lebih kuat terhadap prediksi.'
        )
        return explanation, force_plot_path


class DiabetesBot:
    def __init__(self, token, predictor):
        self.predictor = predictor
        self.application = Application.builder().token(token).build()
        self.application.add_handler(CommandHandler("start", self.start))
        self.application.add_handler(CommandHandler("help", self.help))
        self.application.add_handler(CallbackQueryHandler(self.button))
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.classify))

    async def start(self, update: Update, context: CallbackContext) -> None:
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

    async def help(self, update: Update, context: CallbackContext) -> None:
        help_text = (
            "Instruksi Penggunaan Bot:\n\n"
            "1. Gunakan perintah /start untuk memulai bot.\n"
            "2. Klik tombol 'Mulai Input Data' untuk memasukkan data pasien satu per satu.\n"
            "3. Setelah semua data dimasukkan, bot akan memberikan prediksi apakah pasien tersebut memiliki diabetes atau tidak.\n"
            "4. Gunakan tombol 'Tampilkan Akurasi' untuk melihat akurasi model dan laporan klasifikasi.\n"
            "5. Gunakan tombol 'Tampilkan Grafik Korelasi' untuk melihat grafik korelasi antara fitur-fitur dalam dataset.\n\n"
            "Jika ada pertanyaan atau masalah, hubungi admin."
        )
        await update.message.reply_text(help_text)

    async def button(self, update: Update, context: CallbackContext) -> None:
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
                text=f'**Akurasi model saat ini:** {self.predictor.accuracy:.2f} (atau {self.predictor.accuracy * 100:.2f}%)\n\n'
                     f'**Laporan Klasifikasi:**\n\n'
                     f'{self.predictor.get_classification_report()}',
                parse_mode='Markdown',
                reply_markup=reply_markup
            )
        elif query.data == 'correlation':
            with open(self.predictor.correlation_matrix_path, 'rb') as img:
                await query.message.reply_photo(photo=img)
            await query.edit_message_text(
                text='Berikut adalah grafik korelasi.'
            )

    async def classify(self, update: Update, context: CallbackContext) -> None:
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
                result, input_scaled = self.predictor.predict(user_data)
                await update.message.reply_text(f'Prediksi: {result}')

                try:
                    logging.info(f"Generating SHAP values for input data: {user_data}")
                    explanation, force_plot_path = self.predictor.explain_prediction(input_scaled, user_data)
                    logging.info(f"SHAP values generated successfully")

                    with open(force_plot_path, 'rb') as img:
                        await update.message.reply_document(document=img, filename='shap_force_plot.html', caption=explanation)
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

    def run(self):
        self.application.run_polling()


if __name__ == '__main__':
    data_path = 'diabetes.csv'  # Ganti dengan path dataset Anda
    token = 'Token_bot'  # Ganti dengan token bot Anda
    predictor = DiabetesPredictor(data_path)
    bot = DiabetesBot(token, predictor)
    bot.run()
