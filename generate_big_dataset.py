import pandas as pd
import random

ham_messages = [
    "Bhai kya kar raha hai",
    "Kal milte hain",
    "Assignment complete kiya kya",
    "Mummy ne bulaya hai",
    "Aaj match dekhega kya",
    "Call kar jab free ho",
    "Meeting 5 baje hai",
    "Notes bhej de",
    "College aa raha hai kya",
    "Dinner kar liya kya",
    "Kal exam hai padhai karle",
    "Bhai ghar kab aa raha hai",
    "Aaj movie chalna hai kya",
    "Khana kha liya kya",
    "Project complete ho gaya kya"
]

spam_messages = [
    "Win ₹50000 now click here",
    "Congratulations you won lottery claim now",
    "Free recharge today only",
    "Your account will be blocked update KYC",
    "Click here to verify account immediately",
    "Get 2GB data free today",
    "You won iPhone claim now",
    "Limited offer buy now hurry",
    "Send bank details to receive money",
    "Your OTP expired resend now",
    "Dear customer account suspended verify now",
    "Win cash prize send details",
    "Get 90 percent discount today",
    "Act now limited time offer",
    "Lucky draw winner contact now",
    "Update PAN details to avoid block",
    "Click here to claim refund",
    "SIM will be blocked call now",
    "Free Netflix subscription click now",
    "Instant loan approval apply now",
    "Aapka bank account block hone wala hai update kare",
    "Aapne 25000 jeete hai details bheje",
    "UPI verification ke liye link open kare",
    "Free recharge paane ke liye click kare"
]

data = []

# 🔥 2500 ham + 2500 spam
for _ in range(2500):
    data.append(["ham", random.choice(ham_messages)])

for _ in range(2500):
    data.append(["spam", random.choice(spam_messages)])

# Shuffle data
random.shuffle(data)

df = pd.DataFrame(data, columns=["type", "message"])

# Save
df.to_csv("spam_data.csv", index=False)

print("✅ 5000 dataset created successfully!")