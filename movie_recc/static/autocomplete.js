new autoComplete({
    data: {                              // Data src
      src: films,
    },
    selector: "#autoComplete",           // Input field selector
    threshold: 2,                        // Min. Chars length to start Engine
    debounce: 100,                       // Post duration for engine to start
    searchEngine: "strict",              // Search Engine type/mode
    resultsList: {                       // Rendered results list object
        render: true,
        container: source => {
            source.setAttribute("id", "food_list");
            source.setAttribute("class", "food");
        },
        destination: document.querySelector("#autoComplete"),
        position: "afterend",
        element: "ul"
    },
    maxResults: 5,                         // Max. number of rendered results
    highlight: true,                       // Highlight matching results
    resultItem: {                          // Rendered result item
        content: (data, source) => {
            source.innerHTML = data.match;
            source.setAttribute("class", "food-li");
        },
        element: "li"
    },
    noResults: () => {                     // Action script on noResults
        const result = document.createElement("li");
        result.setAttribute("class", "no_result");
        result.setAttribute("tabindex", "1");
        result.innerHTML = "No Results";
        document.querySelector("#autoComplete_list").appendChild(result);
    },
    onSelection: feedback => {             // Action script onSelection event
        document.getElementById('autoComplete').value = feedback.selection.value;
    }
});
