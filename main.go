package main

import (
	"bufio"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"log"
	"os"
	"path/filepath"

	"github.com/goml/gobrain"
	"github.com/muesli/smartcrop"
	"github.com/muesli/smartcrop/nfnt"
	"github.com/nfnt/resize"
)

func bin(n int) []float64 {
	f := [8]float64{}
	for i := uint(0); i < 8; i++ {
		f[i] = float64((n >> i) & 1)
	}
	return f[:]
}

func dec(d []float64) int {
	n := 0
	for i, v := range d {
		if v > 0.9 {
			n += 1 << uint(i)
		}
	}
	return n
}

func cropImage(img image.Image) image.Image {
	analyzer := smartcrop.NewAnalyzer(nfnt.NewDefaultResizer())
	topCrop, err := analyzer.FindBestCrop(img, 75, 75)
	if err == nil {
		type SubImager interface {
			SubImage(r image.Rectangle) image.Image
		}
		img = img.(SubImager).SubImage(topCrop)
	}
	return resize.Resize(75, 75, img, resize.Lanczos3)
}

func decodeImage(fname string) ([]float64, error) {
	f, err := os.Open(fname)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	src, _, err := image.Decode(f)
	if err != nil {
		return nil, err
	}

	src = cropImage(src)
	bounds := src.Bounds()
	w, h := bounds.Dx(), bounds.Dy()
	if w < h {
		w = h
	} else {
		h = w
	}
	bb := make([]float64, w*h*3)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			r, g, b, _ := src.At(x, y).RGBA()
			bb[y*w*3+x*3] = float64(r) / 255.0
			bb[y*w*3+x*3+1] = float64(g) / 255.0
			bb[y*w*3+x*3+2] = float64(b) / 255.0
		}
	}
	return bb, nil
}

func loadImageSet(category string) ([][]float64, error) {
	result := [][]float64{}
	f, err := os.Open(filepath.Join("dataset", category))
	if err != nil {
		return nil, err
	}
	defer f.Close()
	names, err := f.Readdirnames(-1)
	if err != nil {
		return nil, err
	}
	for _, name := range names {
		fname := filepath.Join("dataset", category, name)
		log.Printf("add %q as %q", fname, category)
		ff, err := decodeImage(fname)
		if err != nil {
			return nil, err
		}
		result = append(result, ff)
	}
	return result, nil
}

func loadModel() (*gobrain.FeedForward, []string, error) {
	f, err := os.Open("labels.txt")
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()

	labels := []string{}
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		labels = append(labels, scanner.Text())
	}
	if scanner.Err() != nil {
		return nil, nil, err
	}

	if len(labels) == 0 {
		return nil, nil, errors.New("No labels found")
	}

	f, err = os.Open("model.json")
	if err != nil {
		return nil, labels, nil
	}
	defer f.Close()

	ff := &gobrain.FeedForward{}
	err = json.NewDecoder(f).Decode(ff)
	if err != nil {
		return nil, labels, err
	}
	return ff, labels, nil
}

func makeModel(labels []string) (*gobrain.FeedForward, error) {
	ff := &gobrain.FeedForward{}
	patterns := [][][]float64{}
	for i, category := range labels {
		bset, err := loadImageSet(category)
		if err != nil {
			return nil, err
		}
		for _, b := range bset {
			patterns = append(patterns, [][]float64{b, bin(i)})
		}
	}
	if len(patterns) == 0 || len(patterns[0][0]) == 0 {
		return nil, errors.New("No images found")
	}
	ff.Init(len(patterns[0][0]), 40, len(patterns[0][1]))
	ff.Train(patterns, 1000, 0.6, 0.4, false)
	return ff, nil
}

func saveModel(ff *gobrain.FeedForward) error {
	f, err := os.Create("model.json")
	if err != nil {
		return err
	}
	defer f.Close()
	return json.NewEncoder(f).Encode(ff)
}

func main() {
	flag.Parse()

	ff, labels, err := loadModel()
	if err != nil {
		log.Fatal(err)
	}
	if ff == nil {
		log.Println("making model file since not found")
		ff, err = makeModel(labels)
		if err != nil {
			log.Fatal(err)
		}
		err = saveModel(ff)
		if err != nil {
			log.Fatal(err)
		}
	}

	if flag.NArg() == 0 {
		flag.Usage()
		os.Exit(2)
	}

	for _, arg := range flag.Args() {
		input, err := decodeImage(arg)
		if err != nil {
			log.Fatal(err)
		}
		n := dec(ff.Update(input))
		if n >= 0 && n < len(labels) {
			fmt.Println(labels[dec(ff.Update(input))])
		} else {
			fmt.Println("unknown image")
		}
	}
}
