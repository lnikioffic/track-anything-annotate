from ui.web.gradio_app import create_app

if __name__ == '__main__':
    print('Starting Gradio annotation app...')

    app = create_app()
    app.launch(
        debug=True,
        share=True,
    )
