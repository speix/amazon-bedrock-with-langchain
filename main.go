package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/tmc/langchaingo/callbacks"
	"github.com/tmc/langchaingo/chains"
	"github.com/tmc/langchaingo/documentloaders"
	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/schema"
	"log"
	"net/http"
)

const (
	format  = "\n\nHuman:%s\n\nAssistant:"
	modelID = "anthropic.claude-v2"
	prompt  = "Give me a summary with maximum of 150 words. Add 3 hashtags at the end to publish on Twitter."
)

type Request struct {
	Prompt            string   `json:"prompt"`
	MaxTokensToSample int      `json:"max_tokens_to_sample"`
	Temperature       float64  `json:"temperature,omitempty"`
	TopP              float64  `json:"top_p,omitempty"`
	TopK              int      `json:"top_k,omitempty"`
	StopSequences     []string `json:"stop_sequences,omitempty"`
}

type Response struct {
	Completion string `json:"completion"`
}

type Model struct {
	CallbacksHandler        callbacks.Handler
	bedrock                 *bedrockruntime.Client
	useHumanAssistantPrompt bool
	modelID                 string
}

func main() {

	large := newLargeLanguageModel()
	chain := chains.LoadStuffQA(large)

	answer, err := chains.Call(context.Background(), chain, map[string]any{
		"input_documents": loadData("https://medium.com/@spei/ai-without-machine-learning-47e90e5ae7c5"),
		"question":        prompt,
	}, chains.WithMaxTokens(500), chains.WithTemperature(0.1))
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(answer["text"].(string))
}

func newLargeLanguageModel() *Model {
	cfg, err := config.LoadDefaultConfig(context.Background())
	if err != nil {
		log.Fatal(err)
	}

	return &Model{
		CallbacksHandler:        nil,
		bedrock:                 bedrockruntime.NewFromConfig(cfg),
		useHumanAssistantPrompt: true,
		modelID:                 modelID,
	}
}

func (m *Model) GeneratePrompt(ctx context.Context, prompts []schema.PromptValue, options ...llms.CallOption) (llms.LLMResult, error) {
	return llms.GeneratePrompt(ctx, m, prompts, options...)
}

func (m *Model) GetNumTokens(text string) int {
	return llms.CountTokens("claude-v2", text)
}

func (m *Model) Call(ctx context.Context, prompt string, options ...llms.CallOption) (string, error) {
	r, err := m.Generate(ctx, []string{prompt}, options...)
	if err != nil {
		return "", err
	}
	if len(r) == 0 {
		return "", errors.New("no response")
	}
	return r[0].Text, nil
}

func (m *Model) Generate(ctx context.Context, prompts []string, options ...llms.CallOption) ([]*llms.Generation, error) {
	if m.CallbacksHandler != nil {
		m.CallbacksHandler.HandleLLMStart(ctx, prompts)
	}

	opts := &llms.CallOptions{}
	for _, opt := range options {
		opt(opts)
	}

	request := Request{
		Prompt:            fmt.Sprintf(format, prompts[0]),
		MaxTokensToSample: opts.MaxTokens,
		Temperature:       opts.Temperature,
		TopK:              opts.TopK,
		TopP:              opts.TopP,
		StopSequences:     opts.StopWords,
	}

	payload, err := json.Marshal(request)
	if err != nil {
		return nil, err
	}

	var resp Response

	resp, err = m.getResponse(payload)
	if err != nil {
		return nil, err
	}

	generations := []*llms.Generation{
		{Text: resp.Completion},
	}

	if m.CallbacksHandler != nil {
		m.CallbacksHandler.HandleLLMEnd(ctx, llms.LLMResult{Generations: [][]*llms.Generation{generations}})
	}
	return generations, nil
}

func loadData(link string) []schema.Document {

	docs, err := getDocsFromLink(link)
	if err != nil {
		log.Fatal(err)
	}

	return docs
}

func getDocsFromLink(link string) ([]schema.Document, error) {
	fmt.Println("loading data from", link)

	resp, err := http.Get(link)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	docs, err := documentloaders.NewHTML(resp.Body).Load(context.Background())
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("successfully loaded data from", link)

	return docs, nil
}

func (m *Model) getResponse(payload []byte) (Response, error) {

	out, err := m.bedrock.InvokeModel(context.Background(), &bedrockruntime.InvokeModelInput{
		Body:        payload,
		ModelId:     aws.String(m.modelID),
		ContentType: aws.String("application/json"),
	})
	if err != nil {
		return Response{}, err
	}
	var resp Response

	err = json.Unmarshal(out.Body, &resp)
	if err != nil {
		return Response{}, err
	}

	return resp, nil
}
