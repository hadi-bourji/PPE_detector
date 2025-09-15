import tkinter as tk
from PIL import Image, ImageTk
import os
import subprocess
from tkinter import messagebox


DATA_DIR = "./PPE_Violation_Images"  # directory on main computer that you want to save different nano image directories under
CAMERAS = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
OBJECTS = ["coat", "gloves", "eyewear", "phone"]

# help load images from folder
def get_images(camera, obj, date):
    folder = os.path.join(DATA_DIR, camera, obj, date)
    if not os.path.exists(folder):
        return []
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith((".png", ".jpg"))]

class PPEViewer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Houston Lab PPE Viewer")
        self.geometry("1000x700")
        self.current_camera = None
        self.current_object = None
        self.current_date = None
        self.image_refs = []  # store PhotoImage refs to prevent garbage collection

        self.container = tk.Frame(self)
        self.container.pack(fill="both", expand=True)

        self.pages = {}
        for Page in (Page1, Page2, Page3):
            page = Page(parent=self.container, controller=self)
            self.pages[Page] = page
            page.grid(row=0, column=0, sticky="nsew")

        self.show_page(Page1)

    def show_page(self, page_class):
        page = self.pages[page_class]
        page.tkraise()


def pull_from_nanos(nanos, local_dir=DATA_DIR):
    for nano in nanos:
        nano_username = nano["username"]
        nano_ip = nano["ip"]
        nano_dir = nano["dir"]
        try:
            subprocess.run(
                ["scp", "-r", f"{nano_username}@{nano_ip}:{nano_dir}", local_dir],
                check=True,
                capture_output=True,
                text=True
            )
            messagebox.showinfo("Success", f"Images pulled successfully from {nano_username}.")
        except subprocess.CalledProcessError as e:
            messagebox.showerror("Error", f"Failed to pull images from {nano_username}:\n{e.stderr}")


class Page1(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        tk.Label(self, text="Select a camera to view.", font=("Arial", 20)).pack(pady=20)

        self.cam_frame = tk.Frame(self)
        self.cam_frame.pack(pady=10)

        self.pull_frame = tk.Frame(self)
        self.pull_frame.pack(anchor="ne", padx=20, pady=10)
        tk.Button(self.pull_frame, text="Retrieve latest images", width=20, command=self.pull_and_refresh).pack()

    def tkraise(self, *args, **kwargs):
        super().tkraise(*args, **kwargs)

        for widget in self.cam_frame.winfo_children():
            widget.destroy()

        global CAMERAS
        CAMERAS = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]

        for cam in CAMERAS:
            cam_label = cam.replace("_", " ").title()
            btn = tk.Button(self.cam_frame, text=cam_label, width=20, command=lambda c=cam: self.select_camera(c))
            btn.pack(pady=5)

    def select_camera(self, camera):
        self.controller.current_camera = camera
        self.controller.show_page(Page2)

    def pull_and_refresh(self):
        nanos = [
            {"ip": "10.172.0.50", "username": "eurofins", "dir": "/home/eurofins/ppe_violations/nano_camera"}
            #add devices here that the main computer will pull images from
        ]
        pull_from_nanos(nanos)
        self.tkraise()



class Page2(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        self.label = tk.Label(self, text="", font=("Arial", 20))
        self.label.pack(pady=20)

        self.sub_label = tk.Label(self, text="Select an object to view violations.", font=("Arial", 16))
        self.sub_label.pack(pady=(0, 15))  

        self.buttons_frame = tk.Frame(self)
        self.buttons_frame.pack(pady=10)

        for obj in OBJECTS:
            obj_label = obj.replace("_", " ").title()
            btn = tk.Button(self.buttons_frame, text=obj_label, width=20, command=lambda o=obj: self.select_object(o))
            btn.pack(pady=5)

        tk.Button(self, text="Back to Cameras", command=lambda: controller.show_page(Page1)).pack(side="bottom", pady=20)

    def tkraise(self, *args, **kwargs):
        super().tkraise(*args, **kwargs)
        camera = self.controller.current_camera
        self.label.config(text=f"Camera: {camera.replace("_", " ").title()}")

    def select_object(self, obj):
        self.controller.current_object = obj
        self.controller.show_page(Page3)

class Page3(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        self.label = tk.Label(self, text="", font=("Arial", 20))
        self.label.pack(pady=10)

        tk.Label(self, text="Select a Date:", font=("Arial", 16)).pack(pady=5)

        # Scrollable listbox for dates
        self.listbox_frame = tk.Frame(self)
        self.listbox_frame.pack(pady=5, fill="x")
        self.date_listbox = tk.Listbox(self.listbox_frame, height=10)
        self.date_scrollbar = tk.Scrollbar(self.listbox_frame, orient="vertical", command=self.date_listbox.yview)
        self.date_listbox.config(yscrollcommand=self.date_scrollbar.set)
        self.date_listbox.pack(side="left", fill="x", expand=True)
        self.date_scrollbar.pack(side="right", fill="y")

        self.date_listbox.bind("<<ListboxSelect>>", self.load_images_from_listbox)
        # Image display area (scrollable)
        self.images_frame = tk.Frame(self)
        self.images_frame.pack(pady=10, fill="both", expand=True)

        # Frame to hold canvas + vertical scrollbar
        self.canvas_frame = tk.Frame(self.images_frame)
        self.canvas_frame.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(self.canvas_frame, height=550, width=950)
        self.scrollbar = tk.Scrollbar(self.canvas_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas)
        self.scrollable_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # Back button stays below the canvas
        tk.Button(self.images_frame, text="Back to Objects", command=lambda: controller.show_page(Page2)).pack(pady=10)

    def tkraise(self, *args, **kwargs):
        super().tkraise(*args, **kwargs)
        cam = self.controller.current_camera
        obj = self.controller.current_object
        self.label.config(text=f"{cam.replace("_", " ").title()} {obj.replace("_", " ")} violations.")

        self.date_listbox.delete(0, tk.END)
        date_dir = os.path.join(DATA_DIR, cam, obj)
        if os.path.exists(date_dir):
            dates = sorted(os.listdir(date_dir))
            for d in dates:
                self.date_listbox.insert(tk.END, d)

        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        self.controller.image_refs.clear()

    def load_images_from_listbox(self, event=None):
        selection = self.date_listbox.curselection()
        if not selection:
            return
        date = self.date_listbox.get(selection[0])
        self.controller.current_date = date
        cam = self.controller.current_camera
        obj = self.controller.current_object

        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        self.controller.image_refs.clear()

        image_paths = get_images(cam, obj, date)
        if not image_paths:
            tk.Label(self.scrollable_frame, text="No images found").pack()
            return
        columns = 3
        for idx, path in enumerate(image_paths):
            try:
                img = Image.open(path)
                img.thumbnail((300, 300))
                photo = ImageTk.PhotoImage(img)
                self.controller.image_refs.append(photo)

                row = idx // columns
                col = idx % columns

                img_label = tk.Label(self.scrollable_frame, image=photo)
                img_label.grid(row=row*2, column=col, padx=5, pady=5)

                text_label = tk.Label(self.scrollable_frame, text=os.path.basename(path))
                text_label.grid(row=row*2+1, column=col, padx=5, pady=(0,10))
            except Exception as e:
                print(f"Error loading {path}: {e}")


if __name__ == "__main__":
    app = PPEViewer()
    app.mainloop()