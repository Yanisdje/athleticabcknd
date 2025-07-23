import OpenAI from "openai";

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

export const analyzeImage = async (imageUrl) => {
  // encode image base64
  const image = await fetch(imageUrl);
  const imageBuffer = await image.arrayBuffer();
  const base64Image = Buffer.from(imageBuffer).toString("base64");

  const response = await openai.images.analyze({
    image: base64Image,
    prompt: `Analyze the image and evaluate the body shape and return a score between 0 and 100 and 
    a description of the body shape.
    with potential improvements.
    
    Your second task should be a workout plan for the body shape. to improve the score given
    `
  });

  return response.data;
};

const sendResponse = async (response) => {
  const response = await openai.chat.completions.create({
    model: "gpt-4o-mini",
    messages: [{ role: "user", content: response }],
  });
  const responseText = response.choices[0].message.content;
  
};

export default openai;