# ==============================================================================
# SCRIPT DE TRANSCRIERE AUDIO ÎN MASĂ PENTRU GOOGLE COLAB (OPTIMIZAT NotebookLM)
# ==============================================================================

# 1. Instalarea automată a dependențelor necesare
# Folosim semnul '!' pentru a rula comenzi de sistem direct din celula Colab
!pip install -qU openai-whisper
!sudo apt-get update -qq
!sudo apt-get install -y -qq ffmpeg

# Importarea bibliotecilor necesare
import os
import torch
import whisper
from google.colab import files

def main():
    # Extensii audio suportate
    extensii_suportate = ('.mp3', '.wav', '.m4a', '.ogg', '.flac')
    nume_fisier_iesire = "transcrieri_cumulate.md"

    # 2. Verificare și configurare Hardware (GPU vs CPU)
    dispozitiv = "cuda" if torch.cuda.is_available() else "cpu"
    print("==================================================")
    print(f"[*] Dispozitiv de procesare detectat: {dispozitiv.upper()}")
    if dispozitiv == "cpu":
        print("[!] AVERTISMENT: Nu s-a detectat un GPU. Transcrierea va fi mai lentă.")
        print("    (Meniu Colab -> Runtime -> Change runtime type -> Hardware accelerator: T4 GPU)")
    print("==================================================\n")

    # 3. Încărcarea modelului Whisper
    print("[*] Se încarcă modelul Whisper 'turbo'...")
    try:
        model = whisper.load_model("turbo", device=dispozitiv)
        print("[*] Model încărcat cu succes!\n")
    except Exception as e:
        print(f"[!] Eroare la încărcarea modelului: {e}")
        return

    # 4. Încărcarea fișierelor audio de către utilizator
    print("[*] Te rog să selectezi fișierele audio din calculatorul tău:")
    fisiere_incarcate = files.upload()
    
    # Filtrarea fișierelor pentru a procesa doar formatele audio suportate
    fisiere_audio = [f for f in fisiere_incarcate.keys() if f.lower().endswith(extensii_suportate)]
    
    if not fisiere_audio:
        print("\n[!] Nu a fost încărcat niciun fișier audio valid. Formate suportate:", extensii_suportate)
        return

    total_fisiere = len(fisiere_audio)
    print(f"\n[*] S-au detectat {total_fisiere} fișier(e) audio pentru procesare.\n")

    # 5. Procesarea fișierelor și crearea fișierului Markdown (.md)
    # Folosim 'w' (write) cu encoding utf-8 pentru a suporta corect diacriticele
    try:
        with open(nume_fisier_iesire, "w", encoding="utf-8") as fisier_md:
            
            for index, nume_fisier in enumerate(fisiere_audio, start=1):
                print(f"-> Se procesează fișierul {index} din {total_fisiere}: '{nume_fisier}'...")
                
                try:
                    # Transcrierea efectivă
                    rezultat = model.transcribe(nume_fisier)
                    text_transcris = rezultat["text"].strip()
                    
                    # 6. Formatare optimizată pentru NotebookLM
                    fisier_md.write(f"# Document: {nume_fisier}\n")
                    fisier_md.write("---\n")
                    fisier_md.write(f"{text_transcris}\n\n\n")
                    
                    print(f"   [+] Transcriere finalizată pentru '{nume_fisier}'.")
                    
                except Exception as e:
                    print(f"   [!] Eroare la transcrierea '{nume_fisier}': {e}")
                
                finally:
                    # 7. Clean-up: Ștergerea fișierului audio temporar pentru a elibera spațiul
                    if os.path.exists(nume_fisier):
                        os.remove(nume_fisier)
                        print(f"   [x] Fișierul '{nume_fisier}' a fost șters de pe server (Clean-up).")

        print("\n==================================================")
        print(f"[*] TOATE FIȘIERELE AU FOST PROCESATE!")
        print(f"[*] Transcrierile au fost salvate în '{nume_fisier_iesire}'.")
        print("==================================================\n")

        # 8. Descărcarea automată a fișierului compilat la final
        print("[*] Se inițiază descărcarea fișierului...")
        files.download(nume_fisier_iesire)

    except Exception as e:
        print(f"[!] A apărut o eroare la crearea fișierului Markdown: {e}")

# Executarea scriptului
if __name__ == "__main__":
    main()
