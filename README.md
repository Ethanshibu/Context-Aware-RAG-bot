# Context-Aware-RAG-bot

This code is an attempt of me trying to implement Retrieval Augmented Generation using the Gemini 1.5 flash model. 

### Progress
- [x] parse through pdfs and chunk them
- [x] vectorise the data onto Chroma DB
- [x] have the llm cite information from uploaded pdf

### Ongoing
- [] implement conversational buffer (memory)
- [] implement persistent file model, making it suitable for deployment
- [] prevent the model from accessing previously uploaded data
- [] implement a method to handle concurrent access by multiple users
- [] implement some form of rate limiting and access (preferably without a database)

