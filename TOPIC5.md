# Get started with generative AI in Azure

In a short few years, generative AI, a subset of artificial intelligence that focuses on creating new content, has changed the way we work and revolutionized what is possible with technology. At times, the fast-moving developments in generative AI can feel challenging to keep track of even for seasoned developers.

In this module, gain a framework for understanding generative AI applications and how Microsoft Azure AI supports innovation. What does today's innovation look like? Consider these use cases:

- *Marketing Content Creation*: Companies use Microsoft Copilot's generative AI to automatically write product descriptions, blog posts, and social media content—saving time and ensuring brand consistency across platforms.

- *Customer Support*: Businesses deploy AI-powered virtual agents that can understand and respond to customer inquiries in natural language, offering 24/7 support and reducing the load on human agents.

- *Code Generation*: Developers use tools like GitHub Copilot to generate code snippets, suggest functions, and even write entire modules based on natural language prompts, speeding up software development.

- *Image and Video Generation*: Designers and content creators use the latest models in Microsoft Foundry's model catalog to generate visuals for campaigns, storyboards, or concept art—often from just a text description.

- *Personalized Learning and Tutoring*: Educational platforms use generative AI to create custom quizzes, explanations, and study guides tailored to a student’s learning style and progress.

Microsoft offers an ecosystem of tools for AI use and development. This module takes a closer look at tools for developers to build generative AI applications. Next, explore the landscape of generative AI applications.

## Understand generative AI applications

Generative AI applications are built with language models. These language models power the 'app logic' component of the interaction between users and generative AI.

https://learn.microsoft.com/en-us/training/wwl-data-ai/get-started-generative-ai-azure/media/application-logic-image.png

### Understand assistants
Generative AI often appears as chat-based assistants that are integrated into applications to help users find information and perform tasks efficiently. One example of such an application is Microsoft Copilot, an AI-powered productivity tool designed to enhance your work experience by providing real-time intelligence and assistance.

### Understand agents
Generative AI that can execute tasks such as filing taxes or coordinating shipping arrangements, just as a few examples, are known as agents. **Agents** are applications that can respond to user input or assess situations autonomously, and take appropriate actions. These actions could help with a series of tasks. For example, an "executive assistant" agent could provide details about the location of a meeting on your calendar, then attach a map or automate the booking of a taxi or rideshare service to help you get there.

Agents contain three main components:

- A language model that powers reasoning and language understanding
- Instructions that define the agent’s goals, behavior, and constraints
- Tools, or functions, that enable the agent to complete tasks

### Use a framework for understanding generative AI applications
One way to think of different generative AI applications is by grouping them in buckets. In general, you can categorize industry and personal generative AI into three buckets, each requiring more customization: ready-to-use applications, extendable applications, and applications you build from the foundation.

1. **Ready-to-use**	These applications are ready-to-use generative AI applications. They do not require any programming work on the user's end to utilize the tool. You can start simply by asking the assistant a question.
2. **Extendable**	Some ready-to-use applications can also be extended using your own data. These customizations enable the assistant to better support specific business processes or tasks. Microsoft Copilot is an example of technology that is ready-to-use and extendable.
3. **Applications you build from the foundation**	You can build your own assistants and assistants with agentic capabilities starting from a language model.

Often, you will use services to extend or build generative AI applications. These services provide the infrastructure, tools, and frameworks necessary to develop, train, and deploy generative AI models. For example, Microsoft provides services such as Copilot Studio to extend Microsoft 365 Copilot and Microsoft Microsoft Foundry to build AI from different models.

## Understand tools to develop generative AI

Microsoft offers a powerful ecosystem of tools and services for building generative AI solutions, designed to support developers, data scientists, and enterprises at every stage of the AI lifecycle. You can develop generative AI solutions with several Microsoft solutions. This module will focus on Microsoft Foundry, Microsoft's unified platform for enterprise AI operations, model builders, and application development.

As a PaaS (platform as a service), Microsoft Foundry gives developers control over the customization of language models used for building applications. These models can be deployed in the cloud and consumed from custom-developed apps and services.

You can use **Microsoft Foundry portal**, a user interface for building, customizing, and managing AI applications and agents—especially those powered by generative AI.

1. **Microsoft Foundry model catalog**	A centralized hub for discovering, comparing, and deploying a wide range of models for generative AI development.
2. **Playgrounds**	Ready-to-use environments for quickly testing ideas, trying out models, and exploring Foundry Models.
3. **Foundry Tools**	In Microsoft Foundry portal, you can build, test, see demos, and deploy Foundry Tools.
4. **Solutions**	You can build agents and customize models in Microsoft Foundry portal.
5. **Observability**	Ability to monitor usage and performance of your application's models.

## Understand Microsoft Foundry's model catalog

Microsoft Foundry provides a comprehensive and dynamic marketplace containing models sold directly by Microsoft and models from its partners and community.

**Azure OpenAI in Foundry models** make up Microsoft's first-party model family and are considered foundation models. Foundation models are pretrained on large texts and can be fine-tuned for specific tasks with a relatively small dataset.

You can deploy the models from Microsoft Foundry model catalog to an endpoint without any extra training. If you want the model to be specialized in a task, or perform better on domain-specific knowledge, you can also choose to customize a foundation model.

To choose the model that best fits your needs, you can test out different models in a playground setting and utilize model leaderboards (preview). Model leaderboards provide a way to see what models are performing best in different criteria such as quality, cost, and throughput. You can also see graphical comparisons of models based on specific metrics.

## Understand Microsoft Foundry capabilities

Microsoft Foundry portal provides a user interface based around hubs and projects. In general, creating a hub provides more comprehensive access to Azure AI and Azure Machine Learning. Within a hub, you can create projects. Projects provide more specific access to models and agent development. You can manage your projects from Microsoft Foundry portal's overview page.

When you create an Azure AI Hub, several other resources are created in tandem, including a Foundry Tools resource. In Microsoft Foundry portal, you can test all kinds of Foundry Tools, including Azure Speech, Azure Language, Azure Vision, and Microsoft Foundry Content Safety.

#### Customizing models
There are many ways to customize the models in generative AI applications. The purpose of customizing your model is to improve aspects of its performance, including quality and safety of the responses. Let's take a look at four of the main ways you can customize models in Microsoft Foundry.

Method	Description
**Using grounding data**	Grounding refers to the process of ensuring that a system's outputs are aligned with factual, contextual, or reliable data sources. Grounding can be done in various ways, such as linking the model to a database, using search engines to retrieve real-time information, or incorporating domain-specific knowledge bases. The goal is to anchor the model's responses to these data sources, enhancing the trustworthiness and applicability of the generated content.
**Implementing Retrieval-Augmented Generation** (RAG)	RAG augments a language model by connecting it to an organization's proprietary database. This technique involves retrieving relevant information from a curated dataset and using it to generate contextually accurate responses. RAG enhances the model's performance by providing it with up-to-date and domain-specific information, which helps in generating more accurate and relevant answers. RAG is useful for applications where real-time access to dynamic data is crucial, such as customer support or knowledge management systems.
**Fine-tuning**	Involves taking a pretrained model and further training it on a smaller, task-specific dataset to make it more suitable for a particular application. This process allows the model to specialize and perform better at specific tasks that require domain-specific knowledge. Fine-tuning is useful for adapting models to domain-specific requirements, improving accuracy, and reducing the likelihood of generating irrelevant or inaccurate responses.
**Managing security and governance controls**	Security and governance controls are needed to manage access, authentication, and data usage. These controls help prevent the publication of incorrect or unauthorized information.

## Understand observability

There are many ways to measure generative AI's response quality. In general, you can think of three dimensions for evaluating and monitoring generative AI. These include:

- **Performance and quality evaluators**: assess the accuracy, groundedness, and relevance of generated content.
- **Risk and safety evaluators**: assess potential risks associated with AI-generated content to safeguard against content risks. This includes evaluating an AI system's predisposition towards generating harmful or inappropriate content.
- **Custom evaluators**: industry-specific metrics to meet specific needs and goals.

Microsoft Foundry supports observability features that improve the performance and trustworthiness of generative AI responses. **Evaluators** are specialized tools in Microsoft Foundry that measure the quality, safety, and reliability of AI responses.

Some evaluators include:

1. Groundedness: measures how consistent the response is with respect to the retrieved context.
2. Relevance: measures how relevant the response is with respect to the query.
3. Fluency: measures natural language quality and readability.
4. Coherence: measures logical consistency and flow of responses.
5. Content safety: comprehensive assessment of various safety concerns.