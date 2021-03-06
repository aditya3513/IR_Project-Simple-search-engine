import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.core.SimpleAnalyzer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.Term;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopScoreDocCollector;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.Version;

/**
 * To create Apache Lucene index in a folder and add files into this index based
 * on the input of the user.
 */
public class Lucene {
    private static Analyzer analyzer = new StandardAnalyzer(Version.LUCENE_47);

    private IndexWriter writer;
    private ArrayList<File> queue = new ArrayList<File>();

    public static void main(String[] args) throws IOException {
      System.out.println("Enter the FULL path where the index will be created: (e.g. /Usr/index or c:\\temp\\index)");

      String indexLocation = null;
      BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
      String s = br.readLine();

      Lucene indexer = null;
      try {
        indexLocation = s;
        indexer = new Lucene(s);
      } catch (Exception ex) {
        System.out.println("Cannot create index..." + ex.getMessage());
        System.exit(-1);
      }

      // ===================================================
      // read input from user until he enters q for quit
      // ===================================================
      while (!s.equalsIgnoreCase("q")) {
        try {
          System.out.println("Enter the FULL path to add into the index (q=quit): (e.g. /home/mydir/docs or c:\\Users\\mydir\\docs)");
          System.out.println("[Acceptable file types: .xml, .html, .html, .txt]");
          s = br.readLine();
          if (s.equalsIgnoreCase("q")) {
            break;
          }

          // try to add file into the index
          indexer.indexFileOrDirectory(s);
        } catch (Exception e) {
          System.out.println("Error indexing " + s + " : " + e.getMessage());
        }
      }

      // ===================================================
      // after adding, we always have to call the
      // closeIndex, otherwise the index is not created
      // ===================================================
      indexer.closeIndex();

      // =========================================================
      // Now search
      // =========================================================
      IndexReader reader = DirectoryReader.open(FSDirectory.open(new File(indexLocation)));
      IndexSearcher searcher = new IndexSearcher(reader);
      //accepting input for path to output file
      System.out.println("Enter the absolute path of output folder");
      String outputFilePath = br.readLine();
      //output file writer
        BufferedWriter writer = new BufferedWriter(new FileWriter(
            outputFilePath + "Top_lucene_docs.txt", true));
        
      System.out.println("Enter the absolute path of search query file");
      String path = br.readLine();
      BufferedReader bufferedReader = new BufferedReader(new FileReader(path));
      String line = bufferedReader.readLine();
      s = "";
      int count = 0;
      while (line != null) {
        try {
          count++;
          String query_id = Integer.toString(count);
          s = line.substring(0, line.length());

          TopScoreDocCollector collector = TopScoreDocCollector.create(100, true);
          Query q = new QueryParser(Version.LUCENE_47, "contents", analyzer).parse(s);
          searcher.search(q, collector);
          ScoreDoc[] hits = collector.topDocs().scoreDocs;

          // 4. display results
          System.out.println("Found " + hits.length + " hits.");
          writer.write(query_id + ") " + s + "\n");

          for (int i = 0; i < hits.length; ++i) {
            int docId = hits[i].doc;
            Document d = searcher.doc(docId);
            String doc_id = d.get("path");
            doc_id = doc_id.substring(doc_id.lastIndexOf("\\") + 1, doc_id.length());
            writer.write(query_id + " Q0 " + doc_id + " " + (i + 1) + " " + hits[i].score + " Lucene");
            writer.newLine();
          }
//          writer.close();
          // 5. term stats --> watch out for which "version" of the term
          // must be checked here instead!
          String[] query_terms = s.split(" ");

          for (int i = 0; i < query_terms.length; i++) {
            Term termInstance = new Term("contents", query_terms[i]);
            long termFreq = reader.totalTermFreq(termInstance);
            long docCount = reader.docFreq(termInstance);
            System.out.println("Term: " + query_terms[i] + " Term Frequency: " + termFreq + ", Document Frequency: " + docCount);
          }
          line = bufferedReader.readLine();
        } catch (Exception e) {
          System.out.println("Error searching " + s + " : " + e.getMessage());
          break;
        }
      }
      writer.close();
      bufferedReader.close();
    }

    /**
     * Constructor
     *
     * @param indexDir the name of the folder in which the index should be created
     * @throws IOException when exception creating index.
     */
    Lucene(String indexDir) throws IOException {

      FSDirectory dir = FSDirectory.open(new File(indexDir));

      IndexWriterConfig config = new IndexWriterConfig(Version.LUCENE_47, analyzer);

      writer = new IndexWriter(dir, config);
    }

    /**
     * Indexes a file or directory
     *
     * @param fileName the name of a text file or a folder we wish to add to the
     *                 index
     * @throws IOException when exception
     */
    public void indexFileOrDirectory(String fileName) throws IOException {
      // ===================================================
      // gets the list of files in a folder (if user has submitted
      // the name of a folder) or gets a single file name (is user
      // has submitted only the file name)
      // ===================================================
      addFiles(new File(fileName));

      int originalNumDocs = writer.numDocs();
      for (File f : queue) {
        FileReader fr = null;
        try {
          Document doc = new Document();

          // ===================================================
          // add contents of file
          // ===================================================
          fr = new FileReader(f);
          doc.add(new TextField("contents", fr));
          doc.add(new StringField("path", f.getPath(), Field.Store.YES));
          doc.add(new StringField("filename", f.getName(), Field.Store.YES));

          writer.addDocument(doc);
          //System.out.println("Added: " + f);
        } catch (Exception e) {
          System.out.println("Could not add: " + f);
        } finally {
          fr.close();
        }
      }

      int newNumDocs = writer.numDocs();
      System.out.println("");
      System.out.println("************************");
      System.out.println((newNumDocs - originalNumDocs) + " documents added.");
      System.out.println("************************");

      queue.clear();
    }

    private void addFiles(File file) {

      if (!file.exists()) {
        System.out.println(file + " does not exist.");
      }
      if (file.isDirectory()) {
        for (File f : file.listFiles()) {
          addFiles(f);
        }
      } else {
        String filename = file.getName().toLowerCase();
        // ===================================================
        // Only index text files
        // ===================================================
        if (filename.endsWith(".htm") || filename.endsWith(".html")
                || filename.endsWith(".xml") || filename.endsWith(".txt")) {
          queue.add(file);
        } else {
          System.out.println("Skipped " + filename);
        }
      }
    }

    /**
     * Close the index.
     *
     * @throws IOException when exception closing
     */
    public void closeIndex() throws IOException {
      writer.close();
    }
}