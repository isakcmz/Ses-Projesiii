import sqlite3
import tkinter as tk
import subprocess     # deneme.py'yi çalıştırmak için
from tkinter import *
from tkinter import messagebox
from tkinter import font


# Veritabanı oluşturma fonksiyonu
def create_user_database():
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            password TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()



# Üye Olma Ekranı
def open_register_screen():
    def register_user():
        username = reg_user.get()
        password = reg_code.get()

        if username == "" or password == "":
            messagebox.showerror("Hata", "Lütfen tüm alanları doldurun.")
            return

        conn = sqlite3.connect("users.db")
        cursor = conn.cursor()

        try:
            cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
            conn.commit()
            messagebox.showinfo("Başarılı", "Kayıt başarılı!")
            register_screen.destroy()
        except sqlite3.IntegrityError:
            messagebox.showerror("Hata", "Kullanıcı adı zaten mevcut.")
        finally:
            conn.close()

    register_screen = Toplevel(app)
    register_screen.title("Üye Ol")
    register_screen.geometry("400x300")
    # Ekranın boyutlarını al
    screen_width = register_screen.winfo_screenwidth()
    screen_height = register_screen.winfo_screenheight()

    # Pencereyi ekranın ortasında konumlandır
    position_top = int(screen_height / 2 - 300 / 2)
    position_left = int(screen_width / 2 - 400 / 2)
    register_screen.geometry(f"400x300+{position_left}+{position_top}")
    register_screen.configure(bg="#fff")
    register_screen.resizable(False, False)

    Label(register_screen, text="Üye Ol", font=("Microsoft YaHei UI Light", 20, "bold"), fg="#1a6fa6", bg="white").pack(pady=20)

    reg_user = Entry(register_screen, width=25, fg="black", border=0, bg="white", font=("Microsoft YaHei UI Light", 11))
    reg_user.pack(pady=10)
    reg_user.insert(0, "Kullanıcı Adı")
    reg_user.bind('<FocusIn>', lambda e: reg_user.delete(0, "end") if reg_user.get() == "Kullanıcı Adı" else None)

    Frame(register_screen, width=250, height=2, bg="black").pack()

    reg_code = Entry(register_screen, width=25, fg="black", border=0, bg="white", font=("Microsoft YaHei UI Light", 11))
    reg_code.pack(pady=10)
    reg_code.insert(0, "Şifre")
    reg_code.bind('<FocusIn>', lambda e: reg_code.delete(0, "end") if reg_code.get() == "Şifre" else None)

    Frame(register_screen, width=250, height=2, bg="black").pack()

    Button(register_screen, text="Kayıt Ol", width=15, pady=7, bg="#2bb27b", fg="white", border=0, command=register_user).pack(pady=20)




# Giriş Yapma Fonksiyonu
def login_user():
    username = user.get()
    password = code.get()

    if username == "" or password == "":
        messagebox.showerror("Hata", "Lütfen tüm alanları doldurun.")
        return

    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password))
    result = cursor.fetchone()

    if result:
        messagebox.showinfo("Başarılı", "Giriş başarılı!")
        app.destroy()
        subprocess.run(["python", "sestanimaprojesi.py"])  # b11.py dosyasını çalıştır
    else:
        messagebox.showerror("Hata", "Kullanıcı adı veya şifre yanlış.")

    conn.close()





def on_enterKullanici(e):
    user.delete(0, "end")

def  on_leaveKullanici(e):
    name = user.get()
    if name == '':
        user.insert(0, "Kullanıcı Adı")

def on_enterSifre(e):
    code.delete(0, "end")

def  on_leaveSifre(e):
    name = code.get()
    if name == '':
        code.insert(0, "Şifre")



app = tk.Tk()
app.title("Giriş Ekranı")
screen_width = app.winfo_screenwidth()
screen_height = app.winfo_screenheight()

# Pencereyi ekranın ortasında konumlandır
position_top = int(screen_height / 2 - 500 / 2)
position_left = int(screen_width / 2 - 900 / 2)
app.geometry(f"900x475+{position_left}+{position_top}")
app.configure(bg="#fff")
app.resizable(False, False)


img =  PhotoImage(file="login.png")
Label(app, image=img, bg="white").place(x=50, y=50)

frame = Frame(app, width=350, height=350, bg="white")
frame.place(x=480, y=70)

heading = Label(app, text="Hoş Geldiniz", fg="#27bbd8", bg="white", font=("Pacifico", 27, "bold"))
heading.place(x=530, y=40)

user = Entry(frame, width=25, fg="black", border=0, bg="white", font=("Microsoft YaHei UI Light", 11))
user.place(x=30, y=80)
user.insert(0, "Kullanıcı Adı")
user.bind('<FocusIn>', on_enterKullanici)
user.bind('<FocusOut>', on_leaveKullanici)

Frame(frame, width=295, height=2, bg="black").place(x=25, y=107)


code = Entry(frame, width=25, fg="black", border=0, bg="white", font=("Microsoft YaHei UI Light", 11))
code.place(x=30, y=150)
code.insert(0, "Şifre")
code.bind('<FocusIn>', on_enterSifre)
code.bind('<FocusOut>', on_leaveSifre)

Frame(frame, width=295, height=2, bg="black").place(x=25, y=177)


Button(frame, width=38, pady=10, text="Giriş Yap", bg="#2bb27b", fg="white", border=0, cursor="hand2", command=login_user).place(x=35, y=204)

label = Label(frame, text="Hesabınız Yok Mu?", fg="black", bg="white", font=("Microsoft YaHei UI Light", 9))
label.place(x=72, y=250)

Uye_ol = Button(frame, width=6, text="Üye Ol", border=0, bg="white", cursor="hand2", fg="#57a1f8",command=open_register_screen)
Uye_ol.place(x=185, y=251)


create_user_database()

app.mainloop()

