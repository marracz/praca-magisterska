package org.mraczynska.recommender.mahout;

import com.google.common.base.Charsets;
import com.google.common.io.Closeables;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.common.iterator.FileLineIterable;

import java.io.*;
import java.util.regex.Pattern;

/**
 * Author: Marta Raczyńska
 */
public class MovieLensDataModel extends FileDataModel {

    private static final Pattern NON_DIGIT_SEMICOLON_PATTERN = Pattern.compile("");

    public MovieLensDataModel(File dataFile) throws IOException {
        super(dataFile);
    }

    public MovieLensDataModel(File dataFile, String delimiterRegex) throws IOException {
        super(dataFile, delimiterRegex);
    }

    public MovieLensDataModel(File dataFile, boolean transpose, long minReloadIntervalMS) throws IOException {
        super(dataFile, transpose, minReloadIntervalMS);
    }

    public MovieLensDataModel(File dataFile, boolean transpose, long minReloadIntervalMS, String delimiterRegex) throws IOException {
        super(dataFile, transpose, minReloadIntervalMS, delimiterRegex);
    }

    public MovieLensDataModel(boolean ignoreRatings) throws IOException {
        this(new File("D:\\recommender\\MovieLens-Ratings.csv"));
    }

    public MovieLensDataModel(File ratingsFile, boolean ignoreRatings) throws IOException {
        super(convertMLFile(ratingsFile, ignoreRatings));
    }

    private static File convertMLFile(File originalFile, boolean ignoreRatings) throws IOException {
        if (!originalFile.exists()) {
            throw new FileNotFoundException(originalFile.toString());
        }
        File resultFile = new File(new File(System.getProperty("java.io.tmpdir")), "taste.movielens.txt");
        resultFile.delete();
        Writer writer = null;
        try {
            writer = new OutputStreamWriter(new FileOutputStream(resultFile), Charsets.UTF_8);
            for (String line : new FileLineIterable(originalFile, true)) {
                // 0 ratings are basically "no rating", ignore them (thanks h.9000)
                if (line.endsWith("\"0\"")) {
                    continue;
                }
                // Delete replace anything that isn't numeric, or a semicolon delimiter. Make comma the delimiter.
                String convertedLine = NON_DIGIT_SEMICOLON_PATTERN.matcher(line)
                        .replaceAll("").replace(';', ',');
                // If this means we deleted an entire ID -- few cases like that -- skip the line
                if (convertedLine.contains(",,")) {
                    continue;
                }
                if (ignoreRatings) {
                    // drop rating
                    convertedLine = convertedLine.substring(0, convertedLine.lastIndexOf(','));
                }
                writer.write(convertedLine);
                writer.write('\n');
            }
            writer.flush();
        } catch (IOException ioe) {
            resultFile.delete();
            throw ioe;
        } finally {
            Closeables.close(writer, false);
        }
        return resultFile;
    }

    @Override
    public String toString() {
        return "MovieLensDataModel";
    }
}
