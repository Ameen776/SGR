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

# ========== قراءة التوكن من متغيرات البيئة ==========
BOT_TOKEN = os.environ.get('BOT_TOKEN')
if not BOT_TOKEN:
    raise ValueError("❌ BOT_TOKEN not found!")

print("✅ Loading data...")

# Load the data
df = pd.read_csv('1XBetCrash.csv')

y = df['Multiplier']
X = df.drop(columns=['Time', 'Multiplier'])

# Convert to numeric if needed
X = X.apply(pd.to_numeric, errors='coerce')
y = pd.to_numeric(y, errors='coerce')

# Drop NaN values
df_clean = pd.concat([X, y], axis=1).dropna()
X = df_clean.drop(columns=['Multiplier'])
y = df_clean['Multiplier']

scaler = StandardScaler()
X = scaler.fit_transform(X)

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=123)

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

print("✅ Models trained successfully!")

# Create bot
bot = telebot.TeleBot(BOT_TOKEN)

@bot.message_handler(commands=['start'])
def handle_start(message: Message):
    bot.reply_to(message, "🎯 مرحباً! أرسل /predict للحصول على توقعات المضاعفات (10 توقعات)")

@bot.message_handler(commands=['predict'])
def handle_predict(message: Message):
    chat_id = message.chat.id
    bot.send_message(chat_id=chat_id, text="🔄 جاري حساب التوقعات...")
    
    for model in [linear_reg, tree_reg, forest_reg, nn_reg]:
        predictions = []
        # استخدام آخر 10 قيم متاحة
        last_X = X[-10:] if len(X) >= 10 else X
        for i, x_val in enumerate(last_X):
            pred = model.predict([x_val])[0]
            predictions.append(f"📊 توقع {i+1}: {pred:.4f}x")
        
        bot.send_message(chat_id=chat_id, text=f"🤖 {model.__class__.__name__}")
        bot.send_message(chat_id=chat_id, text='\n'.join(predictions))
        time.sleep(0.5)

@bot.message_handler(commands=['help'])
def handle_help(message: Message):
    bot.reply_to(message, "📌 الأوامر المتاحة:\n/start - بدء البوت\n/predict - توقعات المضاعفات\n/help - المساعدة")

print("✅ Bot is running... Waiting for commands...")
bot.infinity_polling(timeout=60, long_polling_timeout=60)
