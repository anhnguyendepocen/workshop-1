package nber2;

import java.io.IOException;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;

import java.io.File;
import java.io.FileOutputStream;
import java.net.URL;
import java.nio.channels.Channels;
import java.nio.channels.ReadableByteChannel;

/**
import org.jsoup.HttpStatusException;
import java.io.BufferedWriter;
import java.util.Map;
import java.io.OutputStreamWriter;
import org.jsoup.Connection.Method;
import org.jsoup.Connection.Response;
*/


public class nberclass {

	 public static void main(String[] args) throws IOException {
	        /*
	         * connect to NBER's "Family Members" page and save
	         * the acquired information to the document object "d"
	         */
	        
	    	 Document d = Jsoup.connect("http://www.nber.org/vitae.html")
	                          .userAgent("Mozilla/5.0")
	                          .timeout(15000)
	                          .get();
	        
	        /*
	         * select all "div"-tags directly following the "td" tag
	         * with ID "mainContentTd"
	         */
	        Elements e = d.select("td#mainContentTd > div");
	        
	        /*
	         * let's see what we have selected so far...
	         */
//	        System.out.println(e.html());
	        
	        Elements ePeople = e.select("a[href^=/people]");
	        
//	        System.out.println(ePeople.html());
	        
	        /*
	         * loop over all people:
	         * 
	         * look at everybody's profile page and store their personal
	         * website's URL
	         */
	         
/*
	        for ( Element elPeople : ePeople ) {
*/	        	
	        	
	/****************************************************************
	* DON'T CRASH THE SERVER!
	***************************************************************
	*/
	        	sleep(2000);	                  	
	 /*
	 ***************************************************************
	 * DON'T CRASH THE SERVER! DON'T DELETE "sleep" !!!!!
	 ***************************************************************
	            */

	            
	            /*
	            // print result to screen; comma separated
	            System.out.print(elMember.html() + ",");
	             */
	          
/*	            
	           
	            String Name = elPeople.html();
	            String urlMember = elPeople.absUrl("href");
	          
	        	Document dMember = Jsoup.connect(urlMember)
	                    .userAgent("Mozilla/5.0")
	                    .timeout(30000)
	                    .get();
	            
	        	Elements eMember = dMember.select("p:matchesOwn(.*?E-Mail.*?WWW.*) a[itemprop=url]");
	        	
	        	/*
	        	String eMemberPDF = dMember.select("p:matchesOwn(CV) a[itemprop=url]").attr("abs:href");
	        	download(eMemberPDF, new File(Name + ".pdf"));
	        	*/
	        	
/*	        	
	        	for ( Element elMember : eMember) {
*/	        		
	    
	       /*
	       ****************************************************************
	       * DON'T CRASH THE SERVER!
	 	   ***************************************************************
	 	   */	 	        	
	 	        	sleep(2000);            
	 	    /*
	 	    ***************************************************************
	 	    * DON'T CRASH THE SERVER! DON'T DELETE "sleep" !!!!!
	 	    ***************************************************************
	 	    */
/*	 	        	
	        		try {
	        			String webpage = elMember.absUrl("href");
	        			Document dVita = Jsoup.connect(webpage)
	        			.userAgent("Mozilla/5.0")
	        			.timeout(30000)
	        			.get();
	        			Elements eVita = dVita.select("a:matchesOwn(CV|Vita|Resume|Curriculum Vitae|Curriculum-Vitae|resume|cv|CV|c.v.|c v|C V|C.V.)");        				

	        			for ( Element elVita : eVita ) {
	        			sleep(2000);
	        			String wwwVita = elVita.absUrl("href");
	        			// String pdfVita = elVita.select("matchesOwn(pdf$)").attr("abs:href");
	        			System.out.println(Name + "," + wwwVita);
	        			download (wwwVita, new File(Name + "_p2_")); 
	        			//System.out.println(Name + "," + pdfVita);
	        			
	        			
	        				}
	        		}catch (Exception ex) {}
	        	}
	        }
	 }
*/      	 

	    /*
	     ***************************************************************
	     * VERY IMPORTANT: A PAUSE TO NOT CRASH THE SERVER!!!
	     * DO NOT DELETE!
	     ***************************************************************
	     */
	     
	    private static void sleep(long millis) {
	        try {
	            Thread.sleep(millis);
	        } catch ( InterruptedException ex ) {
	            ex.printStackTrace();
	        }
	    }
	    

	    private static long download(String url, File file) throws IOException {
	        URL content = new URL(url);
	        ReadableByteChannel rbc = Channels.newChannel(content.openStream());
	        FileOutputStream fos = new FileOutputStream(file);
	        try {
	            return fos.getChannel().transferFrom(rbc, 0, Long.MAX_VALUE);
	        } finally {
	            fos.close();
	        }
	    }
	    
}
	        			