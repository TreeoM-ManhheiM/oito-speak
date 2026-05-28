# Rota de inteligência que recebe o áudio
@app.post("/speak")
async def speak_endpoint(audio: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_audio:
        temp_audio.write(await audio.read())
        temp_audio_path = temp_audio.name

    try:
        # 1. Ouvidos: Groq Whisper (Agora em Português)
        with open(temp_audio_path, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(temp_audio_path, file.read()),
                model="whisper-large-v3",
                response_format="json",
                language="pt" # Mudamos de "en" para "pt"
            )
        user_text = transcription.text

        # 2. Cérebro: O Super Professor do ENEM
        system_prompt = """
        Você é um professor particular especialista no ENEM (Exame Nacional do Ensino Médio).
        O aluno vai fazer uma pergunta sobre alguma matéria (História, Matemática, Biologia, etc).
        Sua missão é:
        1. Responder a dúvida de forma clara e direta.
        2. Dar um exemplo rápido de como esse exato assunto costuma "cair" nas questões do ENEM.
        Mantenha a resposta no idioma português do Brasil, seja motivador e aja como um humano. 
        Não use emojis nem formatação de texto complexa, pois sua resposta será lida em voz alta por um sintetizador.
        """

        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text}
            ],
            model="llama-3.3-70b-versatile",
        )
        teacher_response = chat_completion.choices[0].message.content

        return {
            "user_text": user_text, 
            "teacher_response": teacher_response
        }

    finally:
        os.remove(temp_audio_path)
