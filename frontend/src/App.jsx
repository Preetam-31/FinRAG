import { useState } from "react";

import {
  ArrowUpRight,
  Database,
  FileText,
  Search,
  Sparkles,
  ShieldCheck,
  Zap
} from "lucide-react";

import ReactMarkdown from "react-markdown";

import "./App.css";


function App() {

  // =====================================================
  // STATE
  // =====================================================

  const [question, setQuestion] = useState("");

  const [answer, setAnswer] = useState("");

  const [sources, setSources] = useState([]);

  const [loading, setLoading] = useState(false);

  const [error, setError] = useState("");

  const [expandedSource, setExpandedSource] = useState(null);


  // =====================================================
  // ASK QUESTION
  // =====================================================

  const askQuestion = async () => {

    if (!question.trim()) {

      setError("Please enter a question.");

      setAnswer("");

      setSources([]);

      return;
    }


    setLoading(true);

    setAnswer("");

    setSources([]);

    setError("");

    setExpandedSource(null);


    try {

      const response = await fetch(
        "http://127.0.0.1:8000/ask",
        {
          method: "POST",

          headers: {
            "Content-Type": "application/json"
          },

          body: JSON.stringify({
            question: question.trim()
          })
        }
      );


      const data = await response.json();


      if (!response.ok) {

        throw new Error(
          data.detail || "Something went wrong."
        );

      }


      setAnswer(data.answer || "");

      setSources(data.sources || []);

    }


    catch (error) {

      console.error(
        "FinRAG Error:",
        error
      );

      setError(
        error.message ||
        "Unable to connect to the FinRAG backend."
      );

    }


    finally {

      setLoading(false);

    }

  };


  // =====================================================
  // EXAMPLE QUESTIONS
  // =====================================================

  const exampleQuestions = [

    "What was Apple's net income in 2025?",

    "What was Apple's total net sales in 2025?",

    "What was Apple's R&D expense in 2025?",

    "What were Apple's Services net sales in 2025?"

  ];


  // =====================================================
  // HANDLE ENTER KEY
  // =====================================================

  const handleKeyDown = (event) => {

    if (
      event.key === "Enter" &&
      !event.shiftKey
    ) {

      event.preventDefault();

      askQuestion();

    }

  };


  // =====================================================
  // UI
  // =====================================================

  return (

    <div className="app">


      {/* ================================================
          BACKGROUND
      ================================================= */}

      <div className="background-glow glow-one"></div>

      <div className="background-glow glow-two"></div>


      {/* ================================================
          NAVBAR
      ================================================= */}

      <nav className="navbar">

        <div className="brand">

          <div className="brand-icon">

            <Sparkles size={16} />

          </div>

          <span>
            FinRAG
          </span>

        </div>


        <div className="nav-links">

          <a href="#search">
            Ask
          </a>

          <a href="#system">
            System
          </a>

          <a href="#sources">
            Sources
          </a>

          <a href="#about">
            About
          </a>

        </div>


        <div className="nav-status">

          <span className="status-dot"></span>

          System Ready

        </div>

      </nav>


      {/* ================================================
          MAIN
      ================================================= */}

      <main>


        {/* ==============================================
            HERO
        =============================================== */}

        <section
          className="hero"
          id="search"
        >

          <div className="eyebrow-pill">

            <Sparkles size={13} />

            ENTERPRISE FINANCIAL RAG

          </div>


          <h1>

            Ask your financial

            <br />

            <span>
              documents anything.
            </span>

          </h1>


          <p className="hero-description">

            Search, understand, and explore financial reports
            using hybrid retrieval and grounded AI responses.

          </p>


          {/* SEARCH */}

          <div className="search-wrapper">

            <div className="search-box">

              <Search
                size={20}
                className="search-icon"
              />


              <input
                type="text"
                value={question}
                onChange={(event) => {

                  setQuestion(
                    event.target.value
                  );

                  if (error) {
                    setError("");
                  }

                }}
                onKeyDown={handleKeyDown}
                placeholder="Ask a question about the financial report..."
                disabled={loading}
              />


              <button
                className="ask-button"
                onClick={askQuestion}
                disabled={loading}
              >

                {loading ? (

                  <>

                    <span className="button-spinner"></span>

                    Thinking...

                  </>

                ) : (

                  <>

                    Ask

                    <ArrowUpRight size={17} />

                  </>

                )}

              </button>

            </div>

          </div>


          {/* SUGGESTIONS */}

          <div className="suggestions">

            <span className="suggestion-label">
              Try asking
            </span>


            {exampleQuestions.map(
              (item, index) => (

                <button
                  key={index}
                  className="suggestion-button"
                  onClick={() => {

                    setQuestion(item);

                    setError("");

                  }}
                  disabled={loading}
                >

                  {item}

                </button>

              )
            )}

          </div>

        </section>


        {/* ==============================================
            RESPONSE
        =============================================== */}

        <section className="answer-area">


          {/* LOADING */}

          {loading && (

            <div className="response-card loading-card">

              <div className="loading-spinner"></div>


              <div>

                <p className="response-label">
                  FINRAG
                </p>


                <h3>
                  Searching financial documents...
                </h3>


                <p className="loading-description">

                  Running hybrid retrieval,
                  reranking relevant evidence,
                  and generating a grounded response.

                </p>

              </div>

            </div>

          )}


          {/* ERROR */}

          {error && !loading && (

            <div className="response-card error-card">

              <div className="error-symbol">
                !
              </div>


              <div>

                <p className="response-label">
                  REQUEST STATUS
                </p>


                <h3>
                  Unable to generate an answer
                </h3>


                <p className="error-message">
                  {error}
                </p>

              </div>

            </div>

          )}


          {/* SUCCESSFUL ANSWER */}

          {answer && !loading && !error && (

            <div className="response-card answer-card">

              <div className="response-label">
                FINRAG RESPONSE
              </div>


              <h2>
                Answer
              </h2>


              <div className="answer-text">

                <ReactMarkdown>
                  {answer}
                </ReactMarkdown>

              </div>


              {/* ========================================
                  SOURCES
              ========================================= */}

              {sources.length > 0 && (

                <div
                  className="sources"
                  id="sources"
                >

                  <div className="sources-title">

                    <FileText size={16} />

                    <span>
                      Sources
                    </span>

                  </div>


                  <div className="sources-subtitle">

                    Retrieved evidence from the
                    financial document

                  </div>


                  {/* SOURCE CARDS */}

                  {sources.map(
                    (source, index) => (

                      <div
                        className="source-card"
                        key={index}
                      >


                        {/* SOURCE HEADER */}

                        <div className="source-header">


                          <div className="source-page">

                            Page {source.page}

                          </div>


                          {source.score !== undefined && (

                            <div className="source-relevance">

                              RRF Score{" "}

                              {Number(source.score).toFixed(3)}

                            </div>

                          )}

                        </div>


                        {/* SOURCE CONTENT */}

                        <div className="source-content">


                          <p className="source-preview">

                            {expandedSource === index

                              ? source.text

                              : `${source.evidence || source.text.slice(
                                  0,
                                  220
                                )}...`

                            }

                          </p>


                          {/* SHOW / HIDE */}

                          <button
                            className="source-toggle"
                            onClick={() => {

                              if (
                                expandedSource === index
                              ) {

                                setExpandedSource(null);

                              }

                              else {

                                setExpandedSource(index);

                              }

                            }}
                          >

                            {expandedSource === index

                              ? "Hide evidence ↑"

                              : "Show evidence ↓"

                            }

                          </button>

                        </div>

                      </div>

                    )
                  )}

                </div>

              )}

            </div>

          )}

        </section>


        {/* ==============================================
            SYSTEM
        =============================================== */}

        <section
          className="system-section"
          id="system"
        >

          <div className="system-card">


            {/* CARD HEADER */}

            <div className="card-top">

              <div>

                <p className="card-label">
                  RAG SYSTEM
                </p>


                <h2>
                  Built for grounded answers.
                </h2>

              </div>


              <div className="certificate">

                <ShieldCheck size={16} />

                Pipeline Active

              </div>

            </div>


            {/* STATS */}

            <div className="stats-grid">


              <div className="stat">

                <div className="stat-icon">

                  <FileText size={18} />

                </div>


                <div>

                  <strong>
                    80
                  </strong>

                  <span>
                    Pages
                  </span>

                </div>

              </div>


              <div className="stat">

                <div className="stat-icon">

                  <Database size={18} />

                </div>


                <div>

                  <strong>
                    456
                  </strong>

                  <span>
                    Chunks
                  </span>

                </div>

              </div>


              <div className="stat">

                <div className="stat-icon">

                  <Search size={18} />

                </div>


                <div>

                  <strong>
                    Hybrid
                  </strong>

                  <span>
                    Retrieval
                  </span>

                </div>

              </div>


              <div className="stat">

                <div className="stat-icon">

                  <Zap size={18} />

                </div>


                <div>

                  <strong>
                    Top 5
                  </strong>

                  <span>
                    Reranked
                  </span>

                </div>

              </div>

            </div>


            {/* PIPELINE */}

            <div className="pipeline">

              <div className="pipeline-heading">

                <span>
                  RETRIEVAL PIPELINE
                </span>


                <span className="pipeline-active">
                  ACTIVE
                </span>

              </div>


              <div className="pipeline-items">


                <div className="pipeline-item">

                  <span className="pipeline-number">
                    01
                  </span>

                  <span>
                    BM25 Search
                  </span>

                  <i></i>

                </div>


                <div className="pipeline-item">

                  <span className="pipeline-number">
                    02
                  </span>

                  <span>
                    Vector Search
                  </span>

                  <i></i>

                </div>


                <div className="pipeline-item">

                  <span className="pipeline-number">
                    03
                  </span>

                  <span>
                    RRF Fusion
                  </span>

                  <i></i>

                </div>


                <div className="pipeline-item">

                  <span className="pipeline-number">
                    04
                  </span>

                  <span>
                    RRF Ranking
                  </span>

                  <i></i>

                </div>

              </div>

            </div>

          </div>

        </section>


        {/* ==============================================
            KNOWLEDGE BASE
        =============================================== */}

        <section
          className="document-section"
        >

          <div className="document-header">

            <div>

              <p className="card-label">
                KNOWLEDGE BASE
              </p>


              <h2>
                Apple 2025 10-K
              </h2>

            </div>


            <div className="document-badge">

              <span></span>

              Indexed

            </div>

          </div>


          <p className="document-description">

            FinRAG currently uses Apple's 2025 annual
            report as its financial knowledge source.
            Retrieved answers include supporting evidence
            from the indexed document.

          </p>

        </section>

      </main>


      {/* ================================================
          FOOTER
      ================================================= */}

      <footer id="about">

        <div className="footer-brand">

          <div className="brand-icon">

            <Sparkles size={14} />

          </div>


          <span>
            FinRAG
          </span>

        </div>


        <span>
          Hybrid Retrieval · Reranking · Grounded Generation
        </span>


        <span>
          Financial AI System
        </span>

      </footer>

    </div>

  );

}


export default App;