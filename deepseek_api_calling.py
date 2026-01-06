from openai import OpenAI

class DPSKCalling:
    def __init__(self):
        self.client = OpenAI(api_key="sk-5ba90eaf77884677a7b62bede389d38a", base_url="https://api.deepseek.com")

    def create_response(self, content_system, content_user):
        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": content_system},
                {"role": "user", "content": content_user},
            ],
            stream=False
        )
        ret_response = response.choices[0].message.content
        return ret_response

    def __repr__(self):
        return f"client(client={self.client})"


if __name__ == "__main__":
    dpsk_calling = DPSKCalling()
    # content_system = "You are a helpful assistant for Meme's Risk Detection"
    # content_user = "Please analyze the potential risks in the politics, "\
    #                "such as the replacement of specific person's face and family members."\
    #                "Return the values with the structure of the triplet:"\
    #                "<Type, Description of Risk, Potential Feature in the Memes>"
    # content_ret = dpsk_calling.create_response(content_system, content_user)
    # print(content_ret)

    text1 = """
    Artificial intelligence is transforming various industries. 
    Machine learning algorithms can now recognize patterns in large datasets 
    and make predictions with remarkable accuracy.
    """

    text2 = """
    Deep learning models have revolutionized computer vision tasks. 
    These neural networks can identify objects in images almost as well as humans, 
    enabling advancements in autonomous vehicles and medical imaging.
    """

    content_system = "You are a analyzer for the similarity between two texts."
    content_user = (f"Please check whether the following two sentences' topics have similarities."
                    f"You should only return the similarity scores between them.\n"
                    f"Sentence 1: {text1}\n"
                    f"Sentence 2: {text2}\n")
    
    content_ret = dpsk_calling.create_response(content_system, content_user)
    print(content_ret)