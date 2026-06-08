import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import telebot
from telebot.types import Message
import os
import time
import numpy as np

# ========== قراءة التوكن من متغيرات البيئة ==========
BOT_TOKEN = os.environ.get('BOT_TOKEN')
if not BOT_TOKEN:
    raise ValueError("❌ BOT_TOKEN not found in environment variables!")

print("✅ Loading data from 1XBetCrash.csv...")

# Load the data
df = pd.read_csv('1XBetCrash.csv')

# Clean and prepare data
df = df.dropna()
y = df['Multiplier']
X = df.drop(columns=['Time', 'Multiplier'])

# Convert to numeric
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors='coerce')
y = pd.to_numeric(y, errors='coerce')

# Drop any remaining NaN
df_clean = pd.concat([X, y], axis=1).dropna()
X = df_clean.drop(columns=['Multiplier'])
y = df_clean['Multiplier']

if len(X) == 0:
    raise ValueError("❌ No valid data found in CSV file!")

print(f"✅ Data loaded: {len(X)} rows")

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
train_X, test_X, train_y, test_y = train_test_split(X_scaled, y, test_size=0.3, random_state=123)

print("✅ Training models...")

# Train models
linear_reg = LinearRegression()
linear_reg.fit(train_X, train_y)

tree_reg = DecisionTreeRegressor(random_state=123)
tree_reg.fit(train_X, train_y)

forest_reg = RandomForestRegressor(n_estimators=100, random_state=123)
forest_reg.fit(train_X, train_y)

nn_reg = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=123)
nn_reg.fit(train_X, train_y)

print("✅ All models trained successfully!")

# Create bot
bot = telebot.TeleBot(BOT_TOKEN)

@bot.message_handler(commands=['start'])
def handle_start(message: Message):
    bot.reply_to(message, 
        "🎯 *مرحباً بك في بوت توقعات 1xbet Crash!* 🎯\n\n"
        "📊 *الأوامر المتاحة:*\n"
        "/predict - الحصول على توقعات المضاعفات (10 توقعات)\n"
        "/help - عرض المساعدة\n"
        "/about - معلومات عن البوت\n\n"
        "🚀 أرسل /predict لبدء التوقعات!",
        parse_mode='Markdown')

@bot.message_handler(commands=['help'])
def handle_help(message: Message):
    bot.reply_to(message,
        "📌 *كيفية الاستخدام:*\n\n"
        "1️⃣ أرسل /predict للحصول على توقعات المضاعفات\n"
        "2️⃣ ستحصل على 10 توقعات من 4 نماذج ذكاء اصطناعي\n"
        "3️⃣ النماذج: Linear Regression, Decision Tree, Random Forest, Neural Network\n\n"
        "⚠️ *ملاحظة:* التوقعات هي لأغراض تعليمية فقط!",
        parse_mode='Markdown')

@bot.message_handler(commands=['about'])
def handle_about(message: Message):
    bot.reply_to(message,
        "🤖 *بوت توقعات 1xbet Crash*\n\n"
        "📅 الإصدار: 2.0\n"
        "🧠 يعتمد على 4 نماذج تعلم آلي\n"
        "📊 يتم تدريب النماذج على بيانات حقيقية من لعبة Crash\n"
        "👨‍💻 تم تطويره بواسطة @aa-sikkkk\n\n"
        "🔗 GitHub: https://github.com/Ameen776/SGR",
        parse_mode='Markdown')

@bot.message_handler(commands=['predict'])
def handle_predict(message: Message):
    chat_id = message.chat.id
    bot.send_message(chat_id=chat_id, text="🔄 *جاري حساب التوقعات...*\n\n✨ يرجى الانتظار لحظة...", parse_mode='Markdown')
    
    results = []
    
    for model in [linear_reg, tree_reg, forest_reg, nn_reg]:
        predictions = []
        # استخدام آخر 10 قيم متاحة
        last_X = X_scaled[-10:] if len(X_scaled) >= 10 else X_scaled
        for i, x_val in enumerate(last_X):
            pred = model.predict([x_val])[0]
            predictions.append(f"📊 *توقع {i+1}:* `{pred:.4f}x`")
        
        model_name = model.__class__.__name__
        results.append(f"🤖 *{model_name}*\n" + '\n'.join(predictions))
        time.sleep(0.3)
    
    final_message = "🎯 *نتائج التوقعات:*\n\n" + "\n\n".join(results)
    final_message += "\n\n⚠️ *تنبيه:* هذه التوقعات لأغراض تعليمية فقط!"
    
    # تقسيم الرسالة إذا كانت طويلة جداً
    if len(final_message) > 4000:
        for part in results:
            bot.send_message(chat_id=chat_id, text=part, parse_mode='Markdown')
    else:
        bot.send_message(chat_id=chat_id, text=final_message, parse_mode='Markdown')

@bot.message_handler(func=lambda message: True)
def handle_unknown(message: Message):
    bot.reply_to(message, 
        "❓ عذراً، لم أفهم أمرك.\n"
        "📌 الأوامر المتاحة:\n"
        "/start - بدء البوت\n"
        "/predict - توقعات المضاعفات\n"
        "/help - المساعدة\n"
        "/about - معلومات عن البوت")

print("✅ Bot is running... Waiting for commands on Telegram!")
print("="*50)
print("🤖 بوت توقعات 1xbet Crash يعمل بنجاح!")
print("📱 اذهب إلى تليجرام وأرسل /start")
print("="*50)

# Start the bot
bot.infinity_polling(timeout=60, long_polling_timeout=60)